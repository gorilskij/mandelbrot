use crate::rendering::{CoordinatesBox, val_to_color};
use std::num::NonZeroUsize;
use crate::tiles::store::{
    GRID_STRIDES, NUM_GRID_PASSES, NUM_PASSES, TILE_LEN, TILE_SIZE, Tile, TileKey, TileStore,
    depth_for_view, pass_of, pass_pixels, tile_index,
    units_per_pixel,
};
use bytemuck::{Pod, Zeroable};
use dashu::integer::IBig;
use itertools::iproduct;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::window::Window;

const SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VOut {
    return VOut(vec4<f32>(pos, 0.0, 1.0), uv);
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, v.uv);
}
"#;

const BAR_SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VOut {
    return VOut(vec4<f32>(pos, 0.0, 1.0), color);
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
    return v.color;
}
"#;

const PROGRESS_BAR: bool = true;
const BAR_HEIGHT_PX: u32 = 10;
/// Smoothing time of the bar's spring: progress arrives in steps (one per GPU
/// chunk or CPU pass), and jumps back when the view or iteration count
/// changes; the spring turns both into smooth motion.
const BAR_SMOOTH_SECS: f32 = 0.25;

/// The drawn progress bar: a critically damped spring ("SmoothDamp") chasing
/// the real progress in either direction, never overshooting it.
struct BarAnim {
    shown: f32,
    vel:   f32,
    last:  Option<std::time::Instant>,
}

impl BarAnim {
    /// Advance towards `target` (0..=1) and return the fill to draw, or
    /// `None` once complete and settled (bar hidden).
    fn advance(&mut self, target: f32) -> Option<f32> {
        let now = std::time::Instant::now();
        let dt  = self.last.map_or(0.0, |t| (now - t).as_secs_f32()).min(0.1);
        self.last = Some(now);
        self.step(target, dt)
    }

    /// `advance` with an explicit time step.
    fn step(&mut self, target: f32, dt: f32) -> Option<f32> {
        if dt > 0.0 {
            let from   = self.shown;
            let omega  = 2.0 / BAR_SMOOTH_SECS;
            let x      = omega * dt;
            let decay  = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
            let change = self.shown - target;
            let temp   = (self.vel + omega * change) * dt;
            self.vel   = (self.vel - omega * temp) * decay;
            self.shown = target + (change + temp) * decay;
            // Never overshoot: while rising that would show work not yet
            // done; while falling it would bounce.
            if (target > from) == (self.shown > target) {
                self.shown = target;
                self.vel   = 0.0;
            }
        }
        if target >= 1.0 && self.shown >= 0.999 {
            // Done: settle exactly so `animating` stops asking for frames.
            self.shown = 1.0;
            self.vel   = 0.0;
            return None;
        }
        Some(self.shown)
    }

    /// Whether another frame is needed to keep the bar moving.
    fn animating(&self, target: f32) -> bool {
        (target - self.shown).abs() > 1e-4
    }
}

/// How many parent levels to climb looking for a coarser tile to upscale when
/// the target tile has no data yet (the preview while a fresh view renders).
const MAX_CLIMB: usize = 40;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BarVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

struct TileEntry {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    tex_size: u32,
    /// `Tile::version` this texture was built from
    version: u32,
}

pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bar_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    tiles: HashMap<TileKey, TileEntry>,
    bar: BarAnim,
    /// Colour per escape iteration (`val_to_color`), grown on demand: tiles
    /// store iterations and are coloured here, at upload.
    palette: Vec<u32>,
}

impl GpuCompositor {
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .expect("no GPU adapter");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("failed to create device");

            let size = window.inner_size();
            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .find(|f| !f.is_srgb())
                .copied()
                .unwrap_or(caps.formats[0]);

            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            };
            surface.configure(&device, &surface_config);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let bar_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(BAR_SHADER.into()),
            });
            let bar_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });
            let bar_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&bar_layout),
                vertex: wgpu::VertexState {
                    module: &bar_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<BarVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bar_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            GpuCompositor {
                device,
                queue,
                surface,
                surface_config,
                pipeline,
                bar_pipeline,
                bar: BarAnim { shown: 1.0, vel: 0.0, last: None },
                palette: Vec::new(),
                sampler,
                bgl,
                tiles: HashMap::new(),
            }
        })
    }

    /// Force Metal/Vulkan to compile both render pipelines now so the first
    /// real frame doesn't stall. A zero-vertex draw is enough to trigger it.
    pub fn warmup(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..0, 0..1);
            pass.set_pipeline(&self.bar_pipeline);
            pass.draw(0..0, 0..1);
        }
        self.queue.submit([enc.finish()]);
        self.queue.present(frame);
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Draw one frame. Returns whether the progress bar is still moving, i.e.
    /// whether another frame is wanted even if nothing else changes.
    pub fn render(&mut self, store: &TileStore, coords: &CoordinatesBox, width: u32, height: u32) -> bool {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return false,
        };
        let frame_view = frame.texture.create_view(&Default::default());

        let store_frame = store.next_frame();
        let view = coords.view.inner;
        let depth = depth_for_view(view);
        let tile_px = TILE_SIZE as f64 * (units_per_pixel(depth) / view);

        let x0 = tile_index(&coords.origin.x, depth);
        let y0 = tile_index(&coords.origin.y, depth);
        let origin00 = TileKey { depth, x: x0.clone(), y: y0.clone() }.origin();
        let base_sx = (&origin00.x - &coords.origin.x).to_f64().value() / view;
        let base_sy = (&origin00.y - &coords.origin.y).to_f64().value() / view;
        let nx = ((width as f64 - base_sx) / tile_px).ceil().max(1.0) as i64;
        let ny = ((height as f64 - base_sy) / tile_px).ceil().max(1.0) as i64;

        struct Pending {
            src: TileKey,
            px0: f32, py0: f32, px1: f32, py1: f32,
            u0: f32, v0: f32, u1: f32, v1: f32,
        }

        let mut pending: Vec<Pending> = Vec::new();

        for (i, j) in iproduct!(0..nx, 0..ny) {
            let key = TileKey {
                depth,
                x: &x0 + IBig::from(i),
                y: &y0 + IBig::from(j),
            };
            let px0 = (base_sx + i as f64 * tile_px) as f32;
            let py0 = (base_sy + j as f64 * tile_px) as f32;
            let px1 = px0 + tile_px as f32;
            let py1 = py0 + tile_px as f32;

            // Climb ancestors to find best available source.
            let mut candidate = key;
            let mut u0 = 0.0_f32;
            let mut v0 = 0.0_f32;
            let mut frac = 1.0_f32;

            for _ in 0..=MAX_CLIMB {
                if let Some(tile) = store.get(&candidate)
                    && tile.display_passes() > 0
                {
                    tile.touch(store_frame);
                    pending.push(Pending {
                        src: candidate,
                        px0, py0, px1, py1,
                        u0, v0, u1: u0 + frac, v1: v0 + frac,
                    });
                    break;
                }
                let (bx, by) = candidate.parent_offset();
                u0 = u0 / 2.0 + bx as f32 * 0.5;
                v0 = v0 / 2.0 + by as f32 * 0.5;
                frac /= 2.0;
                candidate = candidate.parent();
            }
        }

        // Upload any stale tile textures before recording the render pass.
        for p in &pending {
            self.ensure_texture(&p.src, store);
        }

        // Build one vertex buffer with all quads (6 verts each).
        let w = width as f32;
        let h = height as f32;
        let to_clip = |px: f32, py: f32| -> [f32; 2] {
            [px / w * 2.0 - 1.0, 1.0 - py / h * 2.0]
        };

        let mut verts: Vec<Vertex> = Vec::with_capacity(pending.len() * 6);
        let mut draw_keys: Vec<(&TileKey, u32)> = Vec::with_capacity(pending.len());

        for p in &pending {
            if !self.tiles.contains_key(&p.src) {
                continue;
            }
            let base = verts.len() as u32;
            let [x0, y0] = to_clip(p.px0, p.py0);
            let [x1, y1] = to_clip(p.px1, p.py1);
            let (u0, v0, u1, v1) = (p.u0, p.v0, p.u1, p.v1);
            verts.extend_from_slice(&[
                Vertex { pos: [x0, y0], uv: [u0, v0] },
                Vertex { pos: [x1, y0], uv: [u1, v0] },
                Vertex { pos: [x0, y1], uv: [u0, v1] },
                Vertex { pos: [x1, y0], uv: [u1, v0] },
                Vertex { pos: [x1, y1], uv: [u1, v1] },
                Vertex { pos: [x0, y1], uv: [u0, v1] },
            ]);
            draw_keys.push((&p.src, base));
        }

        // Progress bar: advance the animation and build its geometry.
        let target = bar_progress(store, depth, &x0, &y0, nx, ny);
        let bar = PROGRESS_BAR.then(|| self.bar.advance(target)).flatten().map(|fill| {
            let bar_verts = bar_vertices(height, fill);
            let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&bar_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (vbuf, bar_verts.len() as u32)
        });
        let animating = PROGRESS_BAR && self.bar.animating(target);

        let mut enc = self.device.create_command_encoder(&Default::default());

        if verts.is_empty() {
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                if let Some((vbuf, n)) = &bar {
                    pass.set_pipeline(&self.bar_pipeline);
                    pass.set_vertex_buffer(0, vbuf.slice(..));
                    pass.draw(0..*n, 0..1);
                }
            }
            self.queue.submit([enc.finish()]);
            self.queue.present(frame);
            return animating;
        }

        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            for (key, base) in &draw_keys {
                if let Some(entry) = self.tiles.get(*key) {
                    pass.set_bind_group(0, &entry.bind_group, &[]);
                    pass.draw(*base..*base + 6, 0..1);
                }
            }

            // Progress bar overlay.
            if let Some((vbuf, n)) = &bar {
                pass.set_pipeline(&self.bar_pipeline);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..*n, 0..1);
            }
        }

        self.queue.submit([enc.finish()]);
        self.queue.present(frame);

        // Drop GPU textures for tiles that have been evicted from the CPU store.
        self.tiles.retain(|key, _| store.get(key).is_some());
        animating
    }

    fn ensure_texture(&mut self, key: &TileKey, store: &TileStore) {
        let tile = match store.get(key) {
            Some(t) => t,
            None => return,
        };
        let passes_done = tile.display_passes();
        let version = tile.version();
        if passes_done == 0 {
            return;
        }

        // Grid passes: upload the finest complete grid and let the sampler
        // interpolate. From the first sub-pass on: full resolution, with the
        // missing pixels filled in from their neighbours (`reconstruct`).
        let stride = if passes_done > NUM_GRID_PASSES {
            1
        } else {
            GRID_STRIDES[passes_done as usize - 1]
        };
        let tex_size = (TILE_SIZE / stride) as u32;

        if let Some(entry) = self.tiles.get(key) {
            if entry.version == version {
                return;
            }
            if entry.tex_size == tex_size {
                // Same resolution — overwrite pixels in-place.
                let data = Self::texels(&tile, stride, tex_size, &mut self.palette);
                self.queue.write_texture(
                    entry.texture.as_image_copy(),
                    bytemuck::cast_slice(&data),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(tex_size * 4),
                        rows_per_image: Some(tex_size),
                    },
                    wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
                );
                self.tiles.get_mut(key).unwrap().version = version;
                return;
            }
        }

        // Create a new texture (first upload or size changed due to new pass).
        let data = Self::texels(&tile, stride, tex_size, &mut self.palette);
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&data),
        );
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.tiles.insert(key.clone(), TileEntry { texture, bind_group, tex_size, version });
    }

    /// Texels for a tile displayed at `stride`: the stride grid, or at
    /// stride 1 the reconstructed full-resolution tile.
    /// Texels for a tile displayed at `stride`: the stride grid, or at
    /// stride 1 the reconstructed full-resolution tile.
    fn texels(tile: &Tile, stride: usize, tex_size: u32, palette: &mut Vec<u32>) -> Vec<u32> {
        if stride == 1 {
            Self::reconstruct(tile, palette)
        } else {
            Self::sample_pixels(tile, stride, tex_size, palette)
        }
    }

    /// Sample the stride grid from tile pixels into an Rgba8Unorm buffer.
    /// A missing grid pixel is one being recomputed after the iteration
    /// count went up, i.e. formerly in the set: drawn black, as it was.
    /// Data races are intentional — see store.rs.
    fn sample_pixels(tile: &Tile, stride: usize, tex_size: u32, palette: &mut Vec<u32>) -> Vec<u32> {
        let n = tex_size as usize;
        let mut data = vec![0u32; n * n];
        for row in 0..n {
            for col in 0..n {
                let px = tile.load(row * stride * TILE_SIZE + col * stride).get();
                data[row * n + col] = rgb_to_rgba8(px.map_or(0, |v| color_of(v, palette)));
            }
        }
        data
    }

    /// Full-resolution texels for a tile during (or after) the sub-passes:
    /// computed pixels as they are, each sub-pass gap as the mean colour of
    /// its computed axis neighbours (the sub-pass order guarantees all four
    /// from the first sub-pass on, fewer at the tile edge). A missing pixel
    /// from an already-displayed pass is being recomputed after the
    /// iteration count went up — formerly in the set — and is drawn black.
    fn reconstruct(tile: &Tile, palette: &mut Vec<u32>) -> Vec<u32> {
        let n = TILE_SIZE;
        let shown = tile.display_passes();
        let mut known = |r: usize, c: usize| match tile.load(r * n + c).get() {
            Some(v) => Some(color_of(v, palette)),
            None if pass_of(r, c) < shown => Some(0),
            None => None,
        };
        let mut data = vec![0u32; n * n];
        for r in 0..n {
            for c in 0..n {
                let rgb = match known(r, c) {
                    Some(rgb) => rgb,
                    None => {
                        let neighbours = [
                            (r > 0).then(|| known(r - 1, c)).flatten(),
                            (r + 1 < n).then(|| known(r + 1, c)).flatten(),
                            (c > 0).then(|| known(r, c - 1)).flatten(),
                            (c + 1 < n).then(|| known(r, c + 1)).flatten(),
                        ];
                        let (mut sum, mut k) = ([0u32; 3], 0);
                        for v in neighbours.into_iter().flatten() {
                            for (i, s) in sum.iter_mut().enumerate() {
                                *s += (v >> (8 * i)) & 0xFF;
                            }
                            k += 1;
                        }
                        if k == 0 { 0 } else { (0..3).map(|i| (sum[i] / k) << (8 * i)).sum() }
                    }
                };
                data[r * n + c] = rgb_to_rgba8(rgb);
            }
        }
        data
    }
}

/// Colour (0x00RRGGBB) of a stored pixel value: the escape iteration, 0 for
/// in the set. Looked up in `palette`, which grows as needed.
fn color_of(iteration: u32, palette: &mut Vec<u32>) -> u32 {
    let i = iteration as usize;
    if i >= palette.len() {
        let len = (i + 1).next_power_of_two().max(256);
        palette.extend((palette.len()..len).map(|j| val_to_color(NonZeroUsize::new(j))));
    }
    palette[i]
}

/// 0x00RRGGBB → Rgba8Unorm [R, G, B, 255] as a little-endian u32.
fn rgb_to_rgba8(raw: u32) -> u32 {
    let b = raw & 0xFF;
    let g = (raw >> 8) & 0xFF;
    let r = (raw >> 16) & 0xFF;
    r | (g << 8) | (b << 16) | (0xFF << 24)
}

/// Fraction of the rendering work done for the visible tiles, weighting each
/// finished pass by its share of a tile's pixels (so the bar's sections are
/// proportional to the work). 1.0 when everything is complete.
fn bar_progress(store: &TileStore, depth: i64, x0: &IBig, y0: &IBig, nx: i64, ny: i64) -> f32 {
    let total = (nx * ny) as f32;
    if total == 0.0 {
        return 1.0;
    }
    let done = pass_boundaries();
    let mut sum = 0.0;
    for (i, j) in iproduct!(0..nx, 0..ny) {
        let key = TileKey { depth, x: x0 + IBig::from(i), y: y0 + IBig::from(j) };
        let pd = store.get(&key).map_or(0, |t| t.passes_done()) as usize;
        sum += done[pd.min(NUM_PASSES as usize)];
    }
    sum / total
}

/// Cumulative share of a tile's pixels after each number of finished passes:
/// `[0, 64/16384, 256/16384, …, 1]`.
fn pass_boundaries() -> [f32; NUM_PASSES as usize + 1] {
    let mut out = [0.0; NUM_PASSES as usize + 1];
    for p in 0..NUM_PASSES {
        out[p as usize + 1] = out[p as usize] + pass_pixels(p).count() as f32 / TILE_LEN as f32;
    }
    out
}

/// Bar geometry: a dark track and a solid white fill up to `fill`, across
/// the bottom BAR_HEIGHT_PX pixels.
fn bar_vertices(height: u32, fill: f32) -> Vec<BarVertex> {
    let h = height as f32;
    let y0 = 1.0 - ((height - BAR_HEIGHT_PX) as f32 / h) * 2.0;
    let y1 = -1.0_f32;
    let mut out = Vec::with_capacity(12);
    out.extend(quad([-1.0, y0, 1.0, y1], [0.12, 0.12, 0.12, 1.0]));
    out.extend(quad([-1.0, y0, -1.0 + fill * 2.0, y1], [1.0, 1.0, 1.0, 1.0]));
    out
}

fn quad([x0, y0, x1, y1]: [f32; 4], color: [f32; 4]) -> [BarVertex; 6] {
    let v = |x, y| BarVertex { pos: [x, y], color };
    [v(x0,y0), v(x1,y0), v(x0,y1), v(x1,y0), v(x1,y1), v(x0,y1)]
}

// ---------------------------------------------------------------------------
// Render thread
//
// The compositor owns the wgpu surface, and the only blocking call in the
// whole app — `surface.get_current_texture()` waiting on vsync — lives inside
// `render()`. Running that on the main thread couples frame production to the
// winit event loop: while the main thread is blocked in present, the OS can't
// deliver window-drag/resize events, so dragging stutters whenever the GPU is
// busy.
//
// Moving rendering to its own thread decouples the two. The main thread only
// pumps input and writes the latest view into `Shared`; this thread reads the
// latest view at the top of each loop and draws it. Because it always reads
// *current* state, intermediate frames produced faster than it can draw are
// never encoded — "newest wins", same effect as a Mailbox swapchain but under
// our control and independent of platform support.
// ---------------------------------------------------------------------------

struct Inner {
    coords: CoordinatesBox,
    width: u32,
    height: u32,
    /// bumped whenever the view (coords/size) changes; lets the render thread
    /// tell "nothing changed, park" from "new view, redraw".
    generation: u64,
    /// pending surface reconfigure; only the render thread may touch the surface
    resize: Option<(u32, u32)>,
    exit: bool,
}

struct Shared {
    inner: Mutex<Inner>,
    cv: Condvar,
}

/// Handle to the render thread. The main thread keeps this; the compositor and
/// the actual draw loop live on the spawned thread.
pub struct RenderThread {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl RenderThread {
    pub fn spawn(
        mut compositor: GpuCompositor,
        store: Arc<TileStore>,
        coords: CoordinatesBox,
        width: u32,
        height: u32,
    ) -> Self {
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                coords,
                width,
                height,
                generation: 0,
                resize: None,
                exit: false,
            }),
            cv: Condvar::new(),
        });

        let s = shared.clone();
        let handle = thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                // Compile pipelines up front so the first real frame doesn't stall.
                compositor.warmup();

                loop {
                    // Snapshot the latest view under the lock, then release it
                    // before the (potentially vsync-blocking) render.
                    let (coords, w, h, resize, exit, generation) = {
                        let mut g = s.inner.lock().unwrap();
                        let resize = g.resize.take();
                        (g.coords.clone(), g.width, g.height, resize, g.exit, g.generation)
                    };
                    if exit {
                        break;
                    }
                    if let Some((rw, rh)) = resize {
                        compositor.resize(rw, rh);
                    }

                    // Progress is read *before* rendering: any tiles completed
                    // during the render show up as a change on the next pass and
                    // keep us looping at vsync rate while computation is active.
                    let progress = store.progress();
                    let animating = compositor.render(&store, &coords, w, h);
                    let last_gen = generation;
                    let last_progress = progress;
                    // The progress bar is still moving: draw again right away
                    // (presenting blocks on vsync, so this runs at frame rate).
                    if animating {
                        continue;
                    }

                    // Park until the view changes, a resize is queued, exit is
                    // requested, or computation makes further progress. The
                    // timeout is a safety net for progress bumped asynchronously
                    // by the compute threads (which don't signal the condvar).
                    let mut g = s.inner.lock().unwrap();
                    while !g.exit
                        && g.generation == last_gen
                        && g.resize.is_none()
                        && store.progress() == last_progress
                    {
                        g = s.cv.wait_timeout(g, Duration::from_millis(100)).unwrap().0;
                    }
                }
            })
            .expect("spawn render thread");

        RenderThread { shared, handle: Some(handle) }
    }

    /// Publish a new view for the render thread to draw on its next loop.
    pub fn set_view(&self, coords: CoordinatesBox, width: u32, height: u32) {
        {
            let mut g = self.shared.inner.lock().unwrap();
            g.coords = coords;
            g.width = width;
            g.height = height;
            g.generation = g.generation.wrapping_add(1);
        }
        self.shared.cv.notify_one();
    }

    /// Queue a surface reconfigure (the render thread owns the surface).
    pub fn resize(&self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        {
            let mut g = self.shared.inner.lock().unwrap();
            g.width = width;
            g.height = height;
            g.resize = Some((width, height));
            g.generation = g.generation.wrapping_add(1);
        }
        self.shared.cv.notify_one();
    }

    /// Signal the render thread to stop and wait for it to drop the surface.
    pub fn stop(&mut self) {
        {
            let mut g = self.shared.inner.lock().unwrap();
            g.exit = true;
        }
        self.shared.cv.notify_one();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: f32 = 1.0 / 60.0;

    #[test]
    fn bar_boundaries_are_pass_shares() {
        let b = pass_boundaries();
        assert_eq!(b[0], 0.0);
        assert!((b[NUM_GRID_PASSES as usize] - 0.25).abs() < 1e-6);
        assert!((b[NUM_PASSES as usize] - 1.0).abs() < 1e-6);
        assert!(b.windows(2).all(|w| w[0] < w[1]));
    }

    /// Progress arriving in steps (like GPU chunks) is followed smoothly,
    /// never overtaken, and finishes then hides.
    #[test]
    fn bar_follows_steps_smoothly_and_hides() {
        let mut bar = BarAnim { shown: 0.02, vel: 0.0, last: None };
        let mut target = 0.02_f32;
        let mut prev = 0.02_f32;
        let mut max_step = 0.0_f32;
        for frame in 0..600 {
            if frame % 2 == 0 && target < 1.0 { target = (target + 0.01).min(1.0); } // a chunk every 2 frames
            let Some(shown) = bar.step(target, FRAME) else { break };
            assert!(shown <= target + 1e-6, "ahead of progress");
            assert!(shown >= prev - 1e-6, "went backwards");
            max_step = max_step.max(shown - prev);
            prev = shown;
        }
        // Chunks add 0.01 every 2 frames; the bar moves ~0.005/frame, not in jumps.
        assert!(max_step < 0.0075, "max per-frame step {max_step}");
        // Settles at 100% and hides.
        let mut hidden = false;
        for _ in 0..300 {
            if bar.step(1.0, FRAME).is_none() { hidden = true; break; }
        }
        assert!(hidden);
        assert!(!bar.animating(1.0));
    }

    /// A new view (or iteration change) drops the real progress: the bar
    /// eases down smoothly, never below the new value, and gets there.
    #[test]
    fn bar_eases_down_on_restart() {
        let mut bar = BarAnim { shown: 0.8, vel: 0.3, last: None }; // moving up
        let mut prev = 0.8_f32;
        let mut max_step = 0.0_f32;
        let mut reached = None;
        for frame in 0..120 {
            let shown = bar.step(0.05, FRAME).unwrap();
            assert!(shown >= 0.05 - 1e-6, "undershot");
            max_step = max_step.max((shown - prev).abs());
            prev = shown;
            if reached.is_none() && (shown - 0.05).abs() < 0.005 { reached = Some(frame); }
        }
        let reached = reached.expect("settles at the new progress");
        assert!(reached > 5, "snapped instead of easing ({reached} frames)");
        assert!(reached < 60, "too slow: {reached} frames");
        assert!(max_step < 0.1, "jumped {max_step} in one frame");
    }
}
