use crate::rendering::CoordinatesBox;
use crate::tiles::compose::MAX_CLIMB;
use crate::tiles::store::{
    NUM_PASSES, PASS_STRIDES, TILE_SIZE, Tile, TileKey, TileStore, depth_for_view, tile_index,
    units_per_pixel,
};
use bytemuck::{Pod, Zeroable};
use dashu::integer::IBig;
use itertools::iproduct;
use std::sync::Arc;
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

pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bar_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    // Single composite sample-grid texture (all visible tiles in one texture).
    grid_tex: Option<wgpu::Texture>,
    grid_bg: Option<wgpu::BindGroup>,
    grid_dims: (u32, u32),
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
                    buffers: &[wgpu::VertexBufferLayout {
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
                    }],
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
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
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
                    buffers: &[wgpu::VertexBufferLayout {
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
                    }],
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
                sampler,
                bgl,
                grid_tex: None,
                grid_bg: None,
                grid_dims: (0, 0),
            }
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        // Force grid recreation on next render.
        self.grid_tex = None;
        self.grid_bg = None;
        self.grid_dims = (0, 0);
    }

    pub fn render(&mut self, store: &TileStore, coords: &CoordinatesBox, width: u32, height: u32) {
        // Phase 1: layout
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

        // Phase 3: rebuild sample grid every frame (tiles complete asynchronously).
        {
            let store_frame = store.next_frame();

            // Use the minimum passes_done of visible tiles to set the grid resolution.
            // Each sample becomes one texel; the GPU bilinearly magnifies to screen.
            let mut min_passes = NUM_PASSES as usize + 1;
            for (i, j) in iproduct!(0..nx, 0..ny) {
                let key = TileKey { depth, x: &x0 + IBig::from(i), y: &y0 + IBig::from(j) };
                let pd = store.get(&key).map_or(0, |t| t.passes_done()) as usize;
                if pd > 0 {
                    min_passes = min_passes.min(pd);
                }
            }
            // Fall back to finest resolution if no tile has data yet (ancestor path).
            let grid_tex_size = if min_passes <= NUM_PASSES as usize {
                TILE_SIZE / PASS_STRIDES[min_passes - 1]
            } else {
                TILE_SIZE / PASS_STRIDES[0]
            };

            let grid_w = nx as u32 * grid_tex_size as u32;
            let grid_h = ny as u32 * grid_tex_size as u32;
            let mut buf = vec![0u32; (grid_w * grid_h) as usize];

            for (i, j) in iproduct!(0..nx, 0..ny) {
                let key = TileKey { depth, x: &x0 + IBig::from(i), y: &y0 + IBig::from(j) };
                fill_slot(&mut buf, grid_w, i, j, grid_tex_size, &key, store, store_frame);
            }

            let raw: &[u8] = bytemuck::cast_slice(&buf);

            if self.grid_dims == (grid_w, grid_h) {
                // Same size: overwrite in-place.
                self.queue.write_texture(
                    self.grid_tex.as_ref().unwrap().as_image_copy(),
                    raw,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(grid_w * 4),
                        rows_per_image: Some(grid_h),
                    },
                    wgpu::Extent3d { width: grid_w, height: grid_h, depth_or_array_layers: 1 },
                );
            } else {
                // New size: recreate texture and bind group.
                let tex = self.device.create_texture_with_data(
                    &self.queue,
                    &wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: grid_w, height: grid_h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    raw,
                );
                let tex_view = tex.create_view(&Default::default());
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                self.grid_tex = Some(tex);
                self.grid_bg = Some(bg);
                self.grid_dims = (grid_w, grid_h);
            }

        }

        // Phase 4: draw
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let frame_view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

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

            if let Some(bg) = &self.grid_bg {
                // Fullscreen quad with UV mapping into the sample grid.
                let u0 = (-base_sx / (nx as f64 * tile_px)) as f32;
                let u1 = ((width as f64 - base_sx) / (nx as f64 * tile_px)) as f32;
                let v0 = (-base_sy / (ny as f64 * tile_px)) as f32;
                let v1 = ((height as f64 - base_sy) / (ny as f64 * tile_px)) as f32;

                let verts = [
                    Vertex { pos: [-1.0,  1.0], uv: [u0, v0] },
                    Vertex { pos: [ 1.0,  1.0], uv: [u1, v0] },
                    Vertex { pos: [-1.0, -1.0], uv: [u0, v1] },
                    Vertex { pos: [ 1.0,  1.0], uv: [u1, v0] },
                    Vertex { pos: [ 1.0, -1.0], uv: [u1, v1] },
                    Vertex { pos: [-1.0, -1.0], uv: [u0, v1] },
                ];
                let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..6, 0..1);
            }

            // Progress bar overlay.
            if PROGRESS_BAR {
                if let Some((fill, bg_gray, fill_gray)) = bar_progress(store, depth, &x0, &y0, nx, ny) {
                    let bar_verts = bar_vertices(width, height, fill, bg_gray, fill_gray);
                    let bar_vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&bar_verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    pass.set_pipeline(&self.bar_pipeline);
                    pass.set_vertex_buffer(0, bar_vbuf.slice(..));
                    pass.draw(0..bar_verts.len() as u32, 0..1);
                }
            }
        }

        self.queue.submit([enc.finish()]);
        frame.present();
    }
}

/// Fill one slot of `grid_tex_size × grid_tex_size` texels in the sample-grid
/// buffer for tile `key`. One texel per sample point; the GPU bilinearly
/// magnifies the grid to screen resolution. Data races are intentional.
fn fill_slot(
    buf: &mut [u32],
    grid_w: u32,
    slot_i: i64,
    slot_j: i64,
    grid_tex_size: usize,
    key: &TileKey,
    store: &TileStore,
    frame: u64,
) {
    let mut candidate = key.clone();
    for climb in 0..=MAX_CLIMB {
        if let Some(tile) = store.get(&candidate) {
            let pd = tile.passes_done();
            if pd > 0 {
                tile.touch(frame);
                let stride = PASS_STRIDES[(pd - 1) as usize];
                write_slot(buf, grid_w, slot_i, slot_j, grid_tex_size, &tile, stride, climb, key, &candidate);
                return;
            }
        }
        candidate = candidate.parent();
    }
    // No data available; slot stays black (buf is zero-initialized).
}

/// Write tile data into a `grid_tex_size × grid_tex_size` slot of the
/// sample-grid buffer, optionally extracting an ancestor's sub-region when
/// `climb > 0`. Each output texel maps to the nearest sample point.
fn write_slot(
    buf: &mut [u32],
    grid_w: u32,
    slot_i: i64,
    slot_j: i64,
    grid_tex_size: usize,
    tile: &Tile,
    stride: usize,
    climb: usize,
    key: &TileKey,
    ancestor: &TileKey,
) {
    let n = TILE_SIZE / stride; // tile/ancestor sample count per dimension
    let ts = grid_tex_size as u32;
    let base_row = slot_j as u32 * ts;
    let base_col = slot_i as u32 * ts;

    if climb == 0 {
        for row in 0..ts {
            for col in 0..ts {
                let src_row = row as usize * n / grid_tex_size;
                let src_col = col as usize * n / grid_tex_size;
                let pixel_idx = src_row * stride * TILE_SIZE + src_col * stride;
                let raw = tile.load(pixel_idx).to_raw();
                buf[((base_row + row) * grid_w + base_col + col) as usize] = raw_to_rgba(raw);
            }
        }
    } else {
        let scale = 1usize << climb;
        let ox = usize::try_from(&(&key.x - &ancestor.x * IBig::from(scale as u64))).unwrap();
        let oy = usize::try_from(&(&key.y - &ancestor.y * IBig::from(scale as u64))).unwrap();
        let sub = (n / scale).max(1);
        for row in 0..ts {
            for col in 0..ts {
                let src_row = (oy * sub + row as usize * sub / grid_tex_size).min(n - 1);
                let src_col = (ox * sub + col as usize * sub / grid_tex_size).min(n - 1);
                let pixel_idx = src_row * stride * TILE_SIZE + src_col * stride;
                let raw = tile.load(pixel_idx).to_raw();
                buf[((base_row + row) * grid_w + base_col + col) as usize] = raw_to_rgba(raw);
            }
        }
    }
}

#[inline]
fn raw_to_rgba(raw: u32) -> u32 {
    // 0x00RRGGBB → Rgba8Unorm [R, G, B, 255] as little-endian u32
    let b = raw & 0xFF;
    let g = (raw >> 8) & 0xFF;
    let r = (raw >> 16) & 0xFF;
    r | (g << 8) | (b << 16) | (0xFF << 24)
}

/// Returns (fill 0..1, bg_gray 0..1, fill_gray 0..1) for the progress bar,
/// or None if all tiles have finished all passes (bar should be hidden).
///
/// bg_gray = color of the last fully-completed pass (0 = black if none).
/// fill_gray = color of the pass currently being filled in.
fn bar_progress(
    store: &TileStore,
    depth: i64,
    x0: &IBig,
    y0: &IBig,
    nx: i64,
    ny: i64,
) -> Option<(f32, f32, f32)> {
    let total = (nx * ny) as u32;
    if total == 0 {
        return None;
    }

    // counts[p] = number of tiles with passes_done >= p+1
    let mut counts = [0u32; NUM_PASSES as usize];
    for (i, j) in iproduct!(0..nx, 0..ny) {
        let key = TileKey { depth, x: x0 + IBig::from(i), y: y0 + IBig::from(j) };
        let pd = store.get(&key).map_or(0, |t| t.passes_done()) as usize;
        for p in 0..pd.min(NUM_PASSES as usize) {
            counts[p] += 1;
        }
    }

    // Find the lowest pass that isn't yet complete for all tiles.
    for p in 0..NUM_PASSES as usize {
        if counts[p] < total {
            let fill = counts[p] as f32 / total as f32;
            let bg_gray = p as f32 / NUM_PASSES as f32;
            let fill_gray = (p + 1) as f32 / NUM_PASSES as f32;
            return Some((fill, bg_gray, fill_gray));
        }
    }
    None
}

/// Build bar quad vertices: background (last completed pass color) + colored
/// fill (current pass color), both covering the bottom BAR_HEIGHT_PX pixels.
fn bar_vertices(width: u32, height: u32, fill: f32, bg_gray: f32, fill_gray: f32) -> [BarVertex; 12] {
    let h = height as f32;

    let y0 = 1.0 - ((height - BAR_HEIGHT_PX) as f32 / h) * 2.0;
    let y1 = -1.0_f32;

    let bg_color = [bg_gray, bg_gray, bg_gray, 1.0];
    let fill_color = [fill_gray, fill_gray, fill_gray, 1.0];

    let bg = quad([-1.0, y0, 1.0, y1], bg_color);
    let x1 = -1.0 + fill * 2.0;
    let fg = quad([-1.0, y0, x1, y1], fill_color);

    let mut out = [BarVertex { pos: [0.0; 2], color: [0.0; 4] }; 12];
    out[..6].copy_from_slice(&bg);
    out[6..].copy_from_slice(&fg);
    out
}

fn quad([x0, y0, x1, y1]: [f32; 4], color: [f32; 4]) -> [BarVertex; 6] {
    let v = |x, y| BarVertex { pos: [x, y], color };
    [v(x0,y0), v(x1,y0), v(x0,y1), v(x1,y0), v(x1,y1), v(x0,y1)]
}
