use crate::rendering::{CoordinatesBox, PalettePhase};
use crate::drawing::maybe_pixel::MaybePixel;
use crate::tiles::store::{
    GRID_STRIDES, NUM_GRID_PASSES, NUM_PASSES, TILE_LEN, TILE_SIZE, Tile, TileKey, TileStore,
    pass_of, pass_pixels, tile_index,
    units_per_pixel,
};
use bytemuck::{Pod, Zeroable};
use dashu::integer::IBig;
use itertools::iproduct;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Draws tiles 1:1 into the offscreen image, from the mip level `level` of
/// their layer of a chunk (see `Chunk`).
const SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) level: f32,
    @location(2) @interpolate(flat) layer: u32,
}

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) level: f32,
      @location(3) layer: u32) -> VOut {
    return VOut(vec4<f32>(pos, 0.0, 1.0), uv, level, layer);
}

@group(0) @binding(0) var t: texture_2d_array<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, v.uv, v.layer, v.level);
}
"#;

/// Final pass: area-average the offscreen image down to the screen.
const DOWNSAMPLE_SHADER: &str = include_str!("gpu_compositor_downsample.wgsl");

/// Tile textures and the offscreen image are sRGB, so sampling, filtering and
/// the downsample average in linear light (averaging the gamma-encoded
/// values would darken every blend).
const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Tile colour textures are stored as this and sampled through a
/// `TEXTURE_FORMAT` view: storage textures can't be sRGB, so the colouring
/// pass writes the sRGB encoding itself.
const COLOUR_STORAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Colours tiles from their iteration counts (see the file).
const RECOLOUR_SHADER: &str = include_str!("gpu_compositor_recolour.wgsl");

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
/// Smoothing time of the bar's spring (falling, and rising before the speed
/// is known): progress jumps back when the view or iteration count changes.
const BAR_SMOOTH_SECS: f32 = 0.25;

/// Weight of the newest update in the measured speed and update interval.
const BAR_RATE_ALPHA: f32 = 0.3;

/// Time constant with which the bar's speed follows the wanted speed
/// (momentum: updates change the wanted speed in steps).
const BAR_MOMENTUM_SECS: f32 = 1.0;

/// Wanted lag behind the real progress, in update intervals' worth of
/// progress; deviations are corrected over `BAR_CORRECT_INTERVALS` intervals.
const BAR_LAG_INTERVALS: f32 = 2.0;
const BAR_CORRECT_INTERVALS: f32 = 2.0;

/// Braking: never faster than covering the remaining gap in this time.
const BAR_BRAKE_SECS: f32 = 0.3;

/// The drawn progress bar, never ahead of the real progress. Progress
/// arrives in steps (one per GPU chunk or CPU pass), seconds apart at deep
/// zooms, where a fixed-time spring caught up and then stood still: jumps.
/// So rising, it aims for the measured speed (progress per second over
/// recent updates), corrected to stay about `BAR_LAG_INTERVALS` of an
/// update behind, and its actual speed follows that with momentum
/// (switching speeds at each update was jagged). Falling (a restart), it
/// is a critically damped spring ("SmoothDamp").
struct BarAnim {
    shown: f32,
    vel:   f32,
    last:  Option<std::time::Instant>,
    /// Seconds of animation time (sum of `step`'s dt).
    t:     f32,
    /// Target at the last change, and when it changed (None: not yet seen,
    /// so the first update only starts the clock).
    last_target: f32,
    last_change: Option<f32>,
    /// Measured speed (progress/s) and update interval (s); None until a
    /// rising update has been timed since the last restart.
    rate:     Option<f32>,
    interval: f32,
}

impl BarAnim {
    fn new(shown: f32) -> Self {
        Self {
            shown, vel: 0.0, last: None, t: 0.0,
            last_target: shown, last_change: None, rate: None, interval: BAR_SMOOTH_SECS,
        }
    }

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
        self.t += dt;
        if target > self.last_target + 1e-6 {
            if let Some(since) = self.last_change.map(|c| self.t - c).filter(|&s| s > 0.0) {
                let rate = (target - self.last_target) / since;
                let mix = |old: f32, new: f32| old + BAR_RATE_ALPHA * (new - old);
                self.rate = Some(self.rate.map_or(rate, |r| mix(r, rate)));
                self.interval = mix(self.interval, since);
            }
            self.last_target = target;
            self.last_change = Some(self.t);
        } else if target < self.last_target - 1e-6 {
            // Restart: the old speed no longer applies.
            self.last_target = target;
            self.last_change = Some(self.t);
            self.rate = None;
            self.interval = BAR_SMOOTH_SECS;
            self.vel = 0.0;
        }
        if dt > 0.0 && target >= self.shown && self.rate.is_some() {
            let (rate, interval) = (self.rate.unwrap(), self.interval.max(BAR_SMOOTH_SECS));
            let gap = target - self.shown;
            // Done: no more updates to wait for, so no lag.
            let lag = if target >= 1.0 { 0.0 } else { BAR_LAG_INTERVALS * rate * interval };
            let wanted = (rate + (gap - lag) / (BAR_CORRECT_INTERVALS * interval))
                .max(0.0)
                .min(gap / BAR_BRAKE_SECS);
            self.vel += (wanted - self.vel) * (1.0 - (-dt / BAR_MOMENTUM_SECS).exp());
            self.shown = (self.shown + self.vel.max(0.0) * dt).min(target);
        } else if dt > 0.0 {
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
    /// mip level to sample
    level: f32,
    /// the tile's layer in its chunk
    layer: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BarVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

/// A tile texture's shape: `tex_size`² data texels (the pass-stride grid,
/// or the whole tile), of which the colour mip levels `base..base + levels`
/// are held; finer levels are never drawn at the current s.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Shape {
    tex_size: u32,
    base: u32,
    levels: u32,
}

/// Where a tile's textures live: layer `layer` of chunk `chunk` of the
/// pool for its shape.
#[derive(Clone, Copy, Debug)]
struct Slot {
    shape: Shape,
    chunk: usize,
    layer: u32,
}

struct TileEntry {
    slot: Slot,
    /// `Tile::version` this texture was built from
    version: u32,
    /// `GpuCompositor::palette_gen` it was coloured with
    palette_gen: u32,
}

/// A tile texture to (re)build: see `stale_texture`.
struct Upload {
    key: TileKey,
    tile: Arc<Tile>,
    stride: usize,
    shape: Shape,
    version: u32,
    /// keep the existing slot
    same_shape: bool,
    /// upload the iteration data (else only recolour)
    contents: bool,
}

/// Layers per chunk: the first chunk of a shape is small (most shapes only
/// hold a few tiles for a while), each next one twice the last, up to
/// wgpu's default `max_texture_array_layers`.
const CHUNK_MIN_LAYERS: u32 = 16;
const CHUNK_MAX_LAYERS: u32 = 256;

/// Tiles of one shape, one per layer of 2D array textures: the colouring
/// pass and drawing then take one dispatch / draw call per chunk rather
/// than per tile (Metal serialises dispatches, ~10-15 µs each: a
/// recolour of 24000 tiles took 365 ms as one dispatch per tile, 13-28 ms
/// as one; `diag_recolour_speed`).
struct Chunk {
    /// iteration data (see `iteration_texels`)
    iters: wgpu::Texture,
    /// draws the colour array (sampled as sRGB)
    draw: wgpu::BindGroup,
    /// colouring pass, per colour mip level
    recolour: Vec<wgpu::BindGroup>,
    free: Vec<u32>,
    capacity: u32,
}

/// The image the tiles are drawn into, 1:1 in texels of the mip level in use,
/// before being averaged down to the screen. Sized for the worst case
/// (2 texels per screen pixel per axis, plus a margin) of the current surface.
struct Offscreen {
    view:   wgpu::TextureView,
    width:  u32,
    height: u32,
    /// surface size it was made for
    for_size: (u32, u32),
}

pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bar_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    downsample_bgl: wgpu::BindGroupLayout,
    offscreen: Option<Offscreen>,
    sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
    tiles: HashMap<TileKey, TileEntry>,
    /// chunks per shape (None: dropped once empty)
    pools: HashMap<Shape, Vec<Option<Chunk>>>,
    /// the colouring pass's job lists (layers), `JOB_SLOT` bytes per dispatch
    jobs: wgpu::Buffer,
    bar: BarAnim,
    recolour: Recolour,
    palette_phase: PalettePhase,
    /// the palette phases for the colouring pass (`palette_uniform`)
    palette_buf: wgpu::Buffer,
    /// Bumped on every palette change, so every texture is recoloured.
    palette_gen: u32,
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
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 20,
                                shader_location: 3,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TEXTURE_FORMAT,
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

            // Tiles are drawn 1:1 from an exact mip level (textureSampleLevel),
            // so min/mag only matter for previews magnified from coarser data.
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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

            let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("downsample"),
                source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SHADER.into()),
            });
            let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("downsample"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let downsample_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("downsample"),
                bind_group_layouts: &[Some(&downsample_bgl)],
                immediate_size: 0,
            });
            let downsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("downsample"),
                layout: Some(&downsample_layout),
                vertex: wgpu::VertexState {
                    module: &downsample_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &downsample_shader,
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

            let recolour = Recolour::new(&device);
            let palette_buf = palette_buffer(&device);
            let jobs = jobs_buffer(&device, 64);

            GpuCompositor {
                device,
                queue,
                surface,
                surface_config,
                pipeline,
                bar_pipeline,
                downsample_pipeline,
                downsample_bgl,
                offscreen: None,
                bar: BarAnim::new(1.0),
                palette_phase: PalettePhase::default(),
                palette_buf,
                recolour,
                palette_gen: 0,
                sampler,
                bgl,
                tiles: HashMap::new(),
                pools: HashMap::new(),
                jobs,
            }
        })
    }

    /// Force Metal/Vulkan to compile both render pipelines now so the first
    /// real frame doesn't stall. A zero-vertex draw is enough to trigger it.
    pub fn warmup(&mut self) {
        // Tile and downsample pipelines: an empty tile pass into the
        // offscreen image and a real downsample pass into a scratch target.
        self.ensure_offscreen(self.surface_config.width, self.surface_config.height);
        if let Some(off) = &self.offscreen {
            let tex = |format, usage| self.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let scratch = tex(self.surface_config.format, wgpu::TextureUsages::RENDER_ATTACHMENT);
            let scratch_view = scratch.create_view(&Default::default());
            let dummy_tile = tex(TEXTURE_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING);
            let dummy_tile_view = dummy_tile.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let tile_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dummy_tile_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let downsample_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.downsample_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&off.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            });
            let dummy_verts = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[Vertex { pos: [0.0; 2], uv: [0.0; 2], level: 0.0, layer: 0 }]),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            for (view, tiles) in [(&off.view, true), (&scratch_view, false)] {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                if tiles {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &tile_bg, &[]);
                    pass.set_vertex_buffer(0, dummy_verts.slice(..));
                    pass.draw(0..0, 0..1);
                } else {
                    pass.set_pipeline(&self.downsample_pipeline);
                    pass.set_bind_group(0, &downsample_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            self.queue.submit([enc.finish()]);
        }

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

    /// Recolour with `phase`: every texture is rebuilt on the next frame,
    /// from the tiles' stored iterations (no recomputation).
    pub fn set_palette(&mut self, phase: PalettePhase) {
        if phase != self.palette_phase {
            self.palette_phase = phase;
            self.palette_gen = self.palette_gen.wrapping_add(1);
        }
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
        let depth = store.depth_for_view(view);
        let tile_px = TILE_SIZE as f64 * (units_per_pixel(depth) / view);

        let x0 = tile_index(&coords.origin.x, depth);
        let y0 = tile_index(&coords.origin.y, depth);
        let origin00 = TileKey { depth, x: x0.clone(), y: y0.clone() }.origin();
        let base_sx = (&origin00.x - &coords.origin.x).to_f64().value() / view;
        let base_sy = (&origin00.y - &coords.origin.y).to_f64().value() / view;
        let nx = ((width as f64 - base_sx) / tile_px).ceil().max(1.0) as i64;
        let ny = ((height as f64 - base_sy) / tile_px).ceil().max(1.0) as i64;

        // Antialiasing: a screen pixel spans r = TILE_SIZE / tile_px tile
        // pixels per axis (between s and 2s). Tiles are drawn 1:1 into an
        // offscreen image from mip level k, which leaves q = r / 2^k in
        // [1, 2) offscreen texels per screen pixel (for r < 1: k = 0, q = r),
        // and the downsample pass averages each screen pixel's q×q texels.
        let r = TILE_SIZE as f64 / tile_px;
        let k = mip_level_for(r);
        let q = r / (1u32 << k) as f64;
        let texels_per_tile = (TILE_SIZE >> k) as f64;
        // The offscreen image starts at the texel containing the screen's
        // top-left corner; its texel grid is the level-k tile texel grid.
        let (ox, oy) = ((-base_sx * q).floor(), (-base_sy * q).floor());
        let (frac_x, frac_y) = (-base_sx * q - ox, -base_sy * q - oy);

        struct Pending {
            src: TileKey,
            /// destination rectangle in offscreen texels
            px0: f32, py0: f32, px1: f32, py1: f32,
            u0: f32, v0: f32, u1: f32, v1: f32,
            /// ancestor levels climbed for a preview
            climb: u32,
        }

        let mut pending: Vec<Pending> = Vec::new();

        for (i, j) in iproduct!(0..nx, 0..ny) {
            let key = TileKey {
                depth,
                x: &x0 + IBig::from(i),
                y: &y0 + IBig::from(j),
            };
            let px0 = (i as f64 * texels_per_tile - ox) as f32;
            let py0 = (j as f64 * texels_per_tile - oy) as f32;
            let px1 = px0 + texels_per_tile as f32;
            let py1 = py0 + texels_per_tile as f32;

            // Climb ancestors to find best available source.
            let mut candidate = key;
            let mut u0 = 0.0_f32;
            let mut v0 = 0.0_f32;
            let mut frac = 1.0_f32;

            for climb in 0..=MAX_CLIMB as u32 {
                if let Some(tile) = store.get(&candidate)
                    && tile.display_passes() > 0
                {
                    tile.touch(store_frame);
                    pending.push(Pending {
                        src: candidate,
                        px0, py0, px1, py1,
                        u0, v0, u1: u0 + frac, v1: v0 + frac,
                        climb,
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
        let (k_min, k_max) = drawn_levels(store.min_ratio());
        let k_min = k_min.min(k);
        let span = k_max.saturating_sub(k_min) + 1;
        let mut seen = HashSet::new();
        let uploads: Vec<Upload> = pending
            .iter()
            .filter(|p| seen.insert(&p.src))
            .filter_map(|p| self.stale_texture(&p.src, store, k_min.saturating_sub(p.climb), span))
            .collect();
        let recolour = self.upload_textures(uploads);
        self.ensure_offscreen(width, height);
        let off = self.offscreen.as_ref().expect("offscreen image");

        // One vertex buffer with all quads (6 verts each), in the offscreen
        // image's clip space.
        let (ow, oh) = (off.width as f32, off.height as f32);
        let to_clip = |px: f32, py: f32| -> [f32; 2] {
            [px / ow * 2.0 - 1.0, 1.0 - py / oh * 2.0]
        };

        // Quads grouped by chunk: one draw call per chunk.
        let mut quads: Vec<((Shape, usize), [Vertex; 6])> = Vec::with_capacity(pending.len());
        for p in &pending {
            let Some(entry) = self.tiles.get(&p.src) else { continue };
            let Slot { shape, chunk, layer } = entry.slot;
            // Source texel size in target-depth pixels: the pass stride, times
            // 2 per ancestor level; the level whose texels are 2^k of those
            // is drawn 1:1 (coarser data is magnified from level 0).
            let stride_log2 = (TILE_SIZE / shape.tex_size as usize).trailing_zeros() as i32;
            let level = (k as i32 - stride_log2 - p.climb as i32 - shape.base as i32)
                .clamp(0, shape.levels as i32 - 1) as f32;
            let [x0, y0] = to_clip(p.px0, p.py0);
            let [x1, y1] = to_clip(p.px1, p.py1);
            let (u0, v0, u1, v1) = (p.u0, p.v0, p.u1, p.v1);
            quads.push(((shape, chunk), [
                Vertex { pos: [x0, y0], uv: [u0, v0], level, layer },
                Vertex { pos: [x1, y0], uv: [u1, v0], level, layer },
                Vertex { pos: [x0, y1], uv: [u0, v1], level, layer },
                Vertex { pos: [x1, y0], uv: [u1, v0], level, layer },
                Vertex { pos: [x1, y1], uv: [u1, v1], level, layer },
                Vertex { pos: [x0, y1], uv: [u0, v1], level, layer },
            ]));
        }
        quads.sort_by_key(|(chunk, _)| *chunk);
        let verts: Vec<Vertex> = quads.iter().flat_map(|(_, q)| *q).collect();
        // (chunk, first vertex, vertex count)
        let mut draws: Vec<((Shape, usize), u32, u32)> = Vec::new();
        for (i, (chunk, _)) in quads.iter().enumerate() {
            match draws.last_mut() {
                Some((c, _, n)) if c == chunk => *n += 6,
                _ => draws.push((*chunk, i as u32 * 6, 6)),
            }
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

        // Tiles → offscreen image.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &off.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if !verts.is_empty() {
                let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                for ((shape, chunk), first, n) in &draws {
                    let chunk = self.pools[shape][*chunk].as_ref().expect("chunk in use");
                    pass.set_bind_group(0, &chunk.draw, &[]);
                    pass.draw(*first..*first + *n, 0..1);
                }
            }
        }

        // Offscreen image → screen (area average), then the progress bar.
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[q as f32, 0.0, frac_x as f32, frac_y as f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let downsample_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.downsample_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&off.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
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
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &downsample_bg, &[]);
            pass.draw(0..3, 0..1);

            // Progress bar overlay.
            if let Some((vbuf, n)) = &bar {
                pass.set_pipeline(&self.bar_pipeline);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..*n, 0..1);
            }
        }

        self.queue.submit(recolour.into_iter().chain([enc.finish()]));
        self.queue.present(frame);

        // Free the slots of tiles evicted from the CPU store.
        let evicted: Vec<TileKey> = self.tiles.keys().filter(|key| store.get(key).is_none()).cloned().collect();
        for key in evicted {
            let entry = self.tiles.remove(&key).unwrap();
            self.release(entry.slot);
        }
        animating
    }

    /// (Re)create the offscreen image for a `width`×`height` surface: at most
    /// 2 texels per screen pixel per axis (q < 2), plus 2 for the fractional
    /// start and rounding.
    fn ensure_offscreen(&mut self, width: u32, height: u32) {
        if self.offscreen.as_ref().is_some_and(|o| o.for_size == (width, height)) { return; }
        let max = self.device.limits().max_texture_dimension_2d;
        let (w, h) = ((2 * width + 2).min(max), (2 * height + 2).min(max));
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        log::info!("compositor: offscreen image {w}x{h} for a {width}x{height} surface"); // DIAG
        self.offscreen = Some(Offscreen {
            view: texture.create_view(&Default::default()),
            width: w,
            height: h,
            for_size: (width, height),
        });
    }

    /// What `key`'s tile needs uploaded and recoloured, if it changed (or the palette did),
    /// as the `span` (1 or 2) mip levels drawn when full-resolution texels
    /// are sampled at `level` or up to one coarser (at s = 1 level 0 only; at
    /// s = 3 levels 1 and 2; at s = 4 level 2, a sixteenth of the texels).
    /// Other levels are never drawn at the current s, so they are neither
    /// kept on the GPU nor uploaded; a change of s re-uploads.
    fn stale_texture(&self, key: &TileKey, store: &TileStore, level: u32, span: u32) -> Option<Upload> {
        let tile = store.get(key)?;
        let passes_done = tile.display_passes();
        let version = tile.version();
        if passes_done == 0 {
            return None;
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
        let max_level = tex_size.trailing_zeros();
        let base = level.saturating_sub(stride.trailing_zeros()).min(max_level);
        let levels = (max_level - base + 1).min(span);

        let shape = Shape { tex_size, base, levels };
        let entry = self.tiles.get(key);
        let same_shape = entry.is_some_and(|e| e.slot.shape == shape);
        let contents = !same_shape || entry.is_some_and(|e| e.version != version);
        let recolour = contents || entry.is_some_and(|e| e.palette_gen != self.palette_gen);
        recolour.then(|| Upload { key: key.clone(), tile, stride, shape, version, same_shape, contents })
    }

    /// Bring the stale tiles' textures up to date: upload the iteration
    /// data of those whose contents changed, then colour them all on the GPU
    /// (after a palette change: every tile on screen, from the iteration
    /// data already there), one dispatch per chunk and mip level. Returns
    /// the colouring commands, to be submitted before the frame that draws
    /// them.
    fn upload_textures(&mut self, uploads: Vec<Upload>) -> Option<wgpu::CommandBuffer> {
        if uploads.is_empty() {
            return None;
        }
        self.queue.write_buffer(&self.palette_buf, 0, bytemuck::cast_slice(&palette_uniform(self.palette_phase)));
        let data: Vec<Option<Vec<u32>>> = uploads
            .par_iter()
            .map(|u| u.contents.then(|| iteration_texels(&u.tile, u.stride, u.shape.tex_size)))
            .collect();
        // layers to colour, per chunk
        let mut jobs: HashMap<(Shape, usize), Vec<u32>> = HashMap::new();
        for (u, data) in uploads.into_iter().zip(data) {
            let slot = self.update_entry(u, data);
            jobs.entry((slot.shape, slot.chunk)).or_default().push(slot.layer);
        }
        let mut jobs: Vec<((Shape, usize), Vec<u32>)> = jobs.into_iter().collect();
        jobs.sort_by_key(|(k, _)| *k);

        // One job list per dispatch group, each in its own JOB_SLOT.
        let slot_len = JOB_SLOT as usize / 4;
        let mut lists = vec![0u32; jobs.len() * slot_len];
        for (g, (_, layers)) in jobs.iter().enumerate() {
            lists[g * slot_len..][..layers.len()].copy_from_slice(layers);
        }
        if (lists.len() * 4) as u64 > self.jobs.size() {
            self.jobs = jobs_buffer(&self.device, jobs.len().next_power_of_two());
        }
        self.queue.write_buffer(&self.jobs, 0, bytemuck::cast_slice(&lists));
        let inputs = self.recolour.inputs(&self.device, &self.palette_buf, &self.jobs);

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.recolour.pipeline);
            for (g, ((shape, chunk), layers)) in jobs.iter().enumerate() {
                let chunk = self.pools[shape][*chunk].as_ref().expect("chunk in use");
                pass.set_bind_group(0, &inputs, &[g as u32 * JOB_SLOT]);
                for (i, bg) in chunk.recolour.iter().enumerate() {
                    let groups = (shape.tex_size >> (shape.base + i as u32)).div_ceil(8);
                    pass.set_bind_group(1, bg, &[]);
                    pass.dispatch_workgroups(groups, groups, layers.len() as u32);
                }
            }
        }
        Some(enc.finish())
    }

    /// Give `u`'s tile a slot of its shape (a new one if the shape changed)
    /// and upload its iteration `data` if given (a changed tile, not just a
    /// recolour). Returns the slot.
    fn update_entry(&mut self, u: Upload, data: Option<Vec<u32>>) -> Slot {
        let Upload { key, shape, version, same_shape, .. } = u;
        if !same_shape {
            if let Some(old) = self.tiles.remove(&key) {
                self.release(old.slot);
            }
            let slot = self.alloc(shape);
            self.tiles.insert(key.clone(), TileEntry { slot, version: 0, palette_gen: 0 });
        }
        let entry = self.tiles.get_mut(&key).unwrap();
        let slot = entry.slot;
        if let Some(data) = data {
            let n = shape.tex_size;
            let iters = &self.pools[&shape][slot.chunk].as_ref().expect("chunk in use").iters;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: iters,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot.layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
                wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            );
        }
        entry.version = version;
        entry.palette_gen = self.palette_gen;
        slot
    }

    /// A free slot of `shape`, adding a chunk if all are full.
    fn alloc(&mut self, shape: Shape) -> Slot {
        let chunks = self.pools.entry(shape).or_default();
        if let Some((i, c)) = chunks.iter_mut().enumerate().find_map(|(i, c)| c.as_mut().filter(|c| !c.free.is_empty()).map(|c| (i, c))) {
            return Slot { shape, chunk: i, layer: c.free.pop().unwrap() };
        }
        let held: u32 = chunks.iter().flatten().map(|c| c.capacity).sum();
        let capacity = held.clamp(CHUNK_MIN_LAYERS, CHUNK_MAX_LAYERS);
        let (iters, colour) = chunk_textures(&self.device, shape, capacity);
        let view = draw_view(&colour);
        let draw = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let iters_view = iters.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let recolour = (0..shape.levels).map(|i| self.recolour.tile_group(&self.device, &iters_view, &colour, i)).collect();
        let mut chunk = Chunk { iters, draw, recolour, free: (0..capacity).rev().collect(), capacity };
        let layer = chunk.free.pop().unwrap();
        let chunks = self.pools.get_mut(&shape).unwrap();
        let i = match chunks.iter().position(Option::is_none) {
            Some(i) => { chunks[i] = Some(chunk); i }
            None => { chunks.push(Some(chunk)); chunks.len() - 1 }
        };
        Slot { shape, chunk: i, layer }
    }

    /// Return `slot`; a chunk left empty is dropped.
    fn release(&mut self, slot: Slot) {
        let chunks = self.pools.get_mut(&slot.shape).expect("pool");
        let chunk = chunks[slot.chunk].as_mut().expect("chunk in use");
        chunk.free.push(slot.layer);
        if chunk.free.len() as u32 == chunk.capacity {
            chunks[slot.chunk] = None;
        }
    }
}

/// Iteration data texels for a tile displayed at `stride`: the stride grid,
/// or at stride 1 the whole tile, as `gpu_compositor_recolour.wgsl` takes
/// them. A pixel missing from an already-displayed pass (a grid pixel, or
/// at stride 1 one with `pass_of` < the displayed passes) is being
/// recomputed after the iteration count went up — formerly in the set — so
/// it is 0 (black, as it was); a sub-pass pixel not computed yet stays
/// `MaybePixel::NONE_RAW`, a gap filled from its neighbours.
/// Data races are intentional — see store.rs.
fn iteration_texels(tile: &Tile, stride: usize, tex_size: u32) -> Vec<u32> {
    let n = tex_size as usize;
    let shown = tile.display_passes();
    let mut data = vec![0u32; n * n];
    for r in 0..n {
        for c in 0..n {
            let (tr, tc) = (r * stride, c * stride);
            data[r * n + c] = match tile.load(tr * TILE_SIZE + tc).get() {
                Some(v) => v,
                None if stride > 1 || pass_of(tr, tc) < shown => 0,
                None => MaybePixel::NONE_RAW,
            };
        }
    }
    data
}

#[cfg(test)]
pub fn color_of_for_tests(iteration: u32) -> u32 {
    crate::rendering::val_to_color(std::num::NonZeroUsize::new(iteration as usize), PalettePhase::default())
}

/// Bytes of the jobs buffer per dispatch: a chunk's layers (at most
/// CHUNK_MAX_LAYERS u32), a multiple of the storage offset alignment (256).
const JOB_SLOT: u32 = CHUNK_MAX_LAYERS * 4;

/// A jobs buffer with room for `dispatches` job lists.
fn jobs_buffer(device: &wgpu::Device, dispatches: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("recolour jobs"),
        size: dispatches as u64 * JOB_SLOT as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A chunk's textures for `capacity` tiles of `shape`: the iteration data
/// (R32Uint) and the colour levels (written as `COLOUR_STORAGE_FORMAT`,
/// sampled as `TEXTURE_FORMAT`).
fn chunk_textures(device: &wgpu::Device, shape: Shape, capacity: u32) -> (wgpu::Texture, wgpu::Texture) {
    let n = shape.tex_size;
    let iters = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tile iterations"),
        size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: capacity },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let m = n >> shape.base;
    let colour = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tile colours"),
        size: wgpu::Extent3d { width: m, height: m, depth_or_array_layers: capacity },
        mip_level_count: shape.levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOUR_STORAGE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[TEXTURE_FORMAT],
    });
    (iters, colour)
}

/// The view tiles are drawn from: every layer of a chunk's colour array, as
/// sRGB. Sampling only: the texture also has STORAGE usage (for the
/// colouring pass), which an sRGB view would otherwise inherit and fail.
fn draw_view(colour: &wgpu::Texture) -> wgpu::TextureView {
    colour.create_view(&wgpu::TextureViewDescriptor {
        format: Some(TEXTURE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        ..Default::default()
    })
}

/// The colouring pass (`gpu_compositor_recolour.wgsl`): iteration data →
/// colour texture levels. Group 0: the palette phases and the job lists (a
/// dynamic offset per dispatch); group 1: one chunk's textures, per level.
struct Recolour {
    pipeline: wgpu::ComputePipeline,
    inputs_bgl: wgpu::BindGroupLayout,
    tile_bgl: wgpu::BindGroupLayout,
}

impl Recolour {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recolour"),
            source: wgpu::ShaderSource::Wgsl(RECOLOUR_SHADER.into()),
        });
        let storage = |binding, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let inputs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("recolour inputs"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
            ],
        });
        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("recolour tiles"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: COLOUR_STORAGE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("recolour"),
            bind_group_layouts: &[Some(&inputs_bgl), Some(&tile_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("recolour"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, inputs_bgl, tile_bgl }
    }

    /// Group 0: the palette phases, and a JOB_SLOT window of `jobs`.
    fn inputs(&self, device: &wgpu::Device, palette: &wgpu::Buffer, jobs: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.inputs_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: palette.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: jobs,
                        offset: 0,
                        size: std::num::NonZeroU64::new(JOB_SLOT as u64),
                    }),
                },
            ],
        })
    }

    /// Group 1 for colouring mip level `level` of a chunk's `colour` array
    /// from its `iters` array.
    fn tile_group(&self, device: &wgpu::Device, iters: &wgpu::TextureView, colour: &wgpu::Texture, level: u32) -> wgpu::BindGroup {
        let out = colour.create_view(&wgpu::TextureViewDescriptor {
            format: Some(COLOUR_STORAGE_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(iters) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out) },
            ],
        })
    }
}

/// The palette uniform (`struct Palette` in the colouring shader): the
/// phases, as f32 turns, padded to 16 bytes.
fn palette_uniform(phase: PalettePhase) -> [f32; 4] {
    [phase.hue as f32, phase.light as f32, 0.0, 0.0]
}

/// A buffer for `palette_uniform`.
fn palette_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("palette"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Mip level drawn at r tile pixels per screen pixel: the one leaving
/// q = r / 2^k in [1, 2) (0 for r < 1).
fn mip_level_for(r: f64) -> u32 {
    if r >= 1.0 { (r.log2().floor() as u32).min(TILE_SIZE.trailing_zeros()) } else { 0 }
}

/// Range of mip levels drawn at sampling ratio s, where r is in [s, 2s):
/// floor(log2 s), or one more unless s is a power of two.
fn drawn_levels(s: f64) -> (u32, u32) {
    let lo = mip_level_for(s);
    let hi = mip_level_for(2.0 * s * (1.0 - 1e-9));
    (lo, hi)
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
    palette: PalettePhase,
    /// bumped whenever the view (coords/size) or palette changes; lets the
    /// render thread tell "nothing changed, park" from "new view, redraw".
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
                palette: PalettePhase::default(),
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
                    let (coords, w, h, palette, resize, exit, generation) = {
                        let mut g = s.inner.lock().unwrap();
                        let resize = g.resize.take();
                        (g.coords.clone(), g.width, g.height, g.palette, resize, g.exit, g.generation)
                    };
                    if exit {
                        break;
                    }
                    if let Some((rw, rh)) = resize {
                        compositor.resize(rw, rh);
                    }
                    compositor.set_palette(palette);

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

    /// Recolour with the palette `phase` on the next frame.
    pub fn set_palette(&self, phase: PalettePhase) {
        {
            let mut g = self.shared.inner.lock().unwrap();
            g.palette = phase;
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
    use crate::rendering::val_to_color;
    use std::num::NonZeroUsize;

    const FRAME: f32 = 1.0 / 60.0;

    /// The compositor's WGSL is only compiled when the window opens; check
    /// that it parses and validates.
    #[test]
    fn shaders_validate() {
        for (name, src) in [("tiles", SHADER), ("bar", BAR_SHADER), ("downsample", DOWNSAMPLE_SHADER), ("recolour", RECOLOUR_SHADER)] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    /// Only the levels drawn at s are uploaded: pin the range, and that every
    /// r in [s, 2s) falls inside it.
    #[test]
    fn drawn_levels_cover_the_ratio_range() {
        for (s, want) in [(0.5, (0, 0)), (1.0, (0, 0)), (1.5, (0, 1)), (2.0, (1, 1)),
                          (2.5, (1, 2)), (3.0, (1, 2)), (3.5, (1, 2)), (4.0, (2, 2))] {
            assert_eq!(drawn_levels(s), want, "s = {s}");
            for i in 0..100 {
                let k = mip_level_for(s * (1.0 + i as f64 / 100.0));
                assert!((want.0..=want.1).contains(&k), "s = {s}, k = {k}");
            }
        }
    }

    /// A GPU device for the `--ignored` tests.
    fn test_device() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let adapter = wgpu::Instance::default()
                .request_adapter(&Default::default()).await.expect("no GPU adapter");
            adapter.request_device(&Default::default()).await.expect("no device")
        })
    }

    /// Record the colouring of `layers` of a chunk, one dispatch per level
    /// (as `upload_textures` does), with the job list at offset 0 of `jobs`.
    fn record_recolour(enc: &mut wgpu::CommandEncoder, r: &Recolour, inputs: &wgpu::BindGroup,
                       groups: &[wgpu::BindGroup], shape: Shape, layers: u32) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, inputs, &[0]);
        for (i, bg) in groups.iter().enumerate() {
            let g = (shape.tex_size >> (shape.base + i as u32)).div_ceil(8);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(g, g, layers);
        }
    }

    /// DIAG: time a recolour (palette change) of a whole screen of full
    /// tiles on this machine's GPU: `DIAG_TILES` tiles (default 24000, a
    /// 3000x2000 view at s = 4) drawn from mip level `DIAG_BASE` (default 2),
    /// in chunks of CHUNK_MAX_LAYERS as the compositor does.
    #[test]
    #[ignore]
    fn diag_recolour_speed() {
        let env = |k: &str, d: u32| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
        let (n_tiles, base) = (env("DIAG_TILES", 24000), env("DIAG_BASE", 2));
        let (device, queue) = test_device();
        let r = Recolour::new(&device);
        let palette = palette_buffer(&device);
        queue.write_buffer(&palette, 0, bytemuck::cast_slice(&palette_uniform(PalettePhase::default())));
        let jobs = jobs_buffer(&device, 1);
        queue.write_buffer(&jobs, 0, bytemuck::cast_slice(&(0..CHUNK_MAX_LAYERS).collect::<Vec<u32>>()));
        let inputs = r.inputs(&device, &palette, &jobs);
        let n = TILE_SIZE as u32;
        let shape = Shape { tex_size: n, base, levels: 1 };
        // DIAG_SMOOTH: iterations varying smoothly across the tile, like a
        // real image away from the boundary (default: a different value per
        // texel, the worst case for the palette lookup)
        let smooth = std::env::var("DIAG_SMOOTH").is_ok();
        let data: Vec<u32> = (0..n * n * CHUNK_MAX_LAYERS).map(|i| {
            if smooth { (i % n / 8 + i / n % n / 8 + i / (n * n)) % 2000 } else { (i * 7919) % 2000 }
        }).collect();
        let chunks: Vec<(Vec<wgpu::BindGroup>, u32)> = (0..n_tiles.div_ceil(CHUNK_MAX_LAYERS)).map(|c| {
            let layers = (n_tiles - c * CHUNK_MAX_LAYERS).min(CHUNK_MAX_LAYERS);
            let (iters, colour) = chunk_textures(&device, shape, CHUNK_MAX_LAYERS);
            queue.write_texture(iters.as_image_copy(), bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
                iters.size());
            let view = iters.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default()
            });
            (vec![r.tile_group(&device, &view, &colour, 0)], layers)
        }).collect();
        queue.submit([]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        for round in 0..3 {
            let t = std::time::Instant::now();
            let mut enc = device.create_command_encoder(&Default::default());
            for (groups, layers) in &chunks {
                record_recolour(&mut enc, &r, &inputs, groups, shape, *layers);
            }
            let cmd = enc.finish();
            let encoded = t.elapsed();
            queue.submit([cmd]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            eprintln!("round {round}: {n_tiles} tiles from level {base} in {} chunks: encode {encoded:?}, total {:?}",
                chunks.len(), t.elapsed());
        }
    }

    /// The colouring pass on this machine's GPU (`--ignored`) against the
    /// CPU reference: every texel's colour is `val_to_color` (the shader
    /// computes the palette itself), at several phases and iterations up to
    /// 2^18; mip levels and sub-pass gaps are the mean in linear light of
    /// what they cover (a gap: of its computed axis neighbours, black if
    /// none). The tile sits in layer 2 of a 3-layer chunk. Tolerance: 1 per
    /// channel (f32 vs f64).
    #[test]
    #[ignore]
    fn recolour_matches_cpu() {
        let (device, queue) = test_device();
        let r = Recolour::new(&device);
        let palette = palette_buffer(&device);
        let jobs = jobs_buffer(&device, 1);
        queue.write_buffer(&jobs, 0, bytemuck::cast_slice(&[2u32]));
        let inputs = r.inputs(&device, &palette, &jobs);
        const LAYER: u32 = 2;

        // One 8×8 data tile, drawn at `levels` from `base`: every output byte.
        let run = |data: &[u32], base: u32, levels: u32, phase: PalettePhase| -> Vec<Vec<u8>> {
            queue.write_buffer(&palette, 0, bytemuck::cast_slice(&palette_uniform(phase)));
            let n = 8u32;
            let shape = Shape { tex_size: n, base, levels };
            let (iters, colour) = chunk_textures(&device, shape, LAYER + 1);
            let _draw = draw_view(&colour); // as the compositor makes it
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &iters, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: LAYER }, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
                wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 });
            let view = iters.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default()
            });
            let groups: Vec<_> = (0..levels).map(|i| r.tile_group(&device, &view, &colour, i)).collect();
            let mut enc = device.create_command_encoder(&Default::default());
            record_recolour(&mut enc, &r, &inputs, &groups, shape, 1);
            // read back every level (rows padded to 256 bytes)
            let size = n >> base;
            let bufs: Vec<(wgpu::Buffer, u32)> = (0..levels).map(|i| {
                let m = size >> i;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None, size: 256 * m as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                enc.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo { texture: &colour, mip_level: i, origin: wgpu::Origin3d { x: 0, y: 0, z: LAYER }, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(m) } },
                    wgpu::Extent3d { width: m, height: m, depth_or_array_layers: 1 });
                (buf, m)
            }).collect();
            queue.submit([enc.finish()]);
            bufs.iter().map(|(buf, m)| {
                buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let bytes = buf.slice(..).get_mapped_range().unwrap();
                (0..*m as usize).flat_map(|row| bytes[row * 256..row * 256 + *m as usize * 4].to_vec()).collect()
            }).collect()
        };
        let to_lin = |c: u8| { let c = c as f64 / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } };
        let to_srgb = |l: f64| (255.0 * if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }).round() as i32;
        // expected [r, g, b]: the mean in linear light of the values' colours
        let expect = |vals: &[u32], phase: PalettePhase| -> [i32; 3] {
            let rgb = |v: u32| { let c = val_to_color(NonZeroUsize::new(v as usize), phase); [(c >> 16) as u8, (c >> 8) as u8, c as u8] };
            std::array::from_fn(|ch| to_srgb(vals.iter().map(|&v| to_lin(rgb(v)[ch])).sum::<f64>() / vals.len() as f64))
        };
        let check = |px: &[u8], want: [i32; 3], what: &str| {
            assert_eq!(px[3], 255, "{what}");
            for ch in 0..3 {
                assert!((px[ch] as i32 - want[ch]).abs() <= 1, "{what}: got {:?}, want {want:?}", &px[..3]);
            }
        };

        let phases = [PalettePhase::default(), PalettePhase { hue: 0.3, light: 0.7 }, PalettePhase { hue: 0.93, light: 0.05 }];
        // 64 values: in-set, small, around the default limit, deep
        let vals: Vec<u32> = (0..64u32).map(|i| match i % 4 {
            0 if i == 0 => 0,
            0 => i,
            1 => 2000 + i * 13,
            2 => 30000 + i * 977,
            _ => 262144 - i * 31,
        }).collect();
        for phase in phases {
            let levels = run(&vals, 0, 4, phase);
            for (lvl, out) in levels.iter().enumerate() {
                let (m, f) = (8usize >> lvl, 1usize << lvl);
                for (t, px) in out.chunks(4).enumerate() {
                    let (tr, tc) = (t / m, t % m);
                    let covered: Vec<u32> = (0..f * f).map(|k| vals[(tr * f + k / f) * 8 + tc * f + k % f]).collect();
                    check(px, expect(&covered, phase), &format!("phase {phase:?} level {lvl} texel {t}"));
                }
            }
            // drawn from level 2 only: the 2x2 texels of 4x4 blocks
            for (t, px) in run(&vals, 2, 1, phase)[0].chunks(4).enumerate() {
                let covered: Vec<u32> = (0..16).map(|k| vals[((t / 2) * 4 + k / 4) * 8 + (t % 2) * 4 + k % 4]).collect();
                check(px, expect(&covered, phase), &format!("phase {phase:?} base 2 texel {t}"));
            }
        }
        // a gap between two in-set and two escaped neighbours; a gap with no
        // computed neighbour: black
        let v = 1234;
        let mut gaps = vec![0u32; 64];
        gaps[3 * 8 + 4] = v; gaps[5 * 8 + 4] = v;
        gaps[4 * 8 + 4] = MaybePixel::NONE_RAW;
        gaps[0] = MaybePixel::NONE_RAW; gaps[1] = MaybePixel::NONE_RAW; gaps[8] = MaybePixel::NONE_RAW;
        let l0 = &run(&gaps, 0, 1, PalettePhase::default())[0];
        check(&l0[(4 * 8 + 4) * 4..][..4], expect(&[0, 0, v, v], PalettePhase::default()), "gap");
        check(&l0[0..4], [0, 0, 0], "gap without neighbours");
    }

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
        let mut bar = BarAnim::new(0.02);
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

    /// Progress arriving seconds apart (deep zooms): once the speed is
    /// measured the bar keeps moving between updates at about that speed,
    /// instead of catching up and standing still.
    #[test]
    fn bar_moves_steadily_between_slow_updates() {
        let mut bar = BarAnim::new(0.0);
        let mut target = 0.0_f32;
        let (mut prev, mut still, mut max_step) = (0.0_f32, 0, 0.0_f32);
        for frame in 0..600 {
            if frame % 90 == 0 { target = (target + 0.1).min(1.0); } // every 1.5 s
            let shown = bar.step(target, FRAME).unwrap();
            assert!(shown <= target + 1e-6, "ahead of progress");
            if frame >= 270 { // after a few updates
                if shown - prev < 1e-5 { still += 1; }
                max_step = max_step.max(shown - prev);
            }
            prev = shown;
        }
        // Steady speed is 0.1 per 90 frames ≈ 0.0011/frame.
        assert!(still < 30, "stood still for {still} of 330 frames");
        assert!(max_step < 0.004, "max per-frame step {max_step}");
    }

    /// Irregular updates (uneven intervals and sizes, like real chunks and
    /// glitch rounds): the bar's per-frame movement changes gradually, not
    /// in hiccups at each update.
    #[test]
    fn bar_is_smooth_with_irregular_updates() {
        let mut bar = BarAnim::new(0.0);
        let intervals = [40, 120, 70, 150, 55, 100, 80, 130];
        let sizes = [0.02, 0.07, 0.03, 0.08, 0.04, 0.05, 0.05, 0.06];
        let (mut target, mut next, mut k) = (0.0_f32, 0, 0);
        let mut moves = vec![];
        let mut prev = 0.0_f32;
        for frame in 0..1500 {
            if frame == next && target < 1.0 {
                target = (target + sizes[k % sizes.len()]).min(1.0);
                next += intervals[k % intervals.len()];
                k += 1;
            }
            let Some(shown) = bar.step(target, FRAME) else { break };
            assert!(shown <= target + 1e-6, "ahead of progress");
            if frame > 300 { moves.push(shown - prev); }
            prev = shown;
        }
        let mean = moves.iter().sum::<f32>() / moves.len() as f32;
        let jerk = moves.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max) / mean;
        let still = moves.iter().filter(|&&m| m < 0.1 * mean).count();
        eprintln!("mean move {mean:.5}/frame, max change {jerk:.2}x mean, nearly still {still}/{}", moves.len());
        assert!(jerk < 0.15, "per-frame movement jumps by {jerk:.2}x its mean");
        assert!(still * 10 < moves.len(), "nearly still for {still} frames");
    }

    /// A new view (or iteration change) drops the real progress: the bar
    /// eases down smoothly, never below the new value, and gets there.
    #[test]
    fn bar_eases_down_on_restart() {
        let mut bar = BarAnim::new(0.8);
        bar.vel = 0.3; // moving up
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
