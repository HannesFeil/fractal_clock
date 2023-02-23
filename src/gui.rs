#![allow(clippy::extra_unused_type_parameters)]

use std::ops::{Add, Mul};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages,
};
use winit::{dpi::PhysicalSize, window::Window};

const RECURSION_DEPTH: usize = 12;
const WORK_GROUP_INITIAL_RECURSION_DEPTH: usize = 7;
const COMPUTE_RECURSION_DEPTH: usize = RECURSION_DEPTH - WORK_GROUP_INITIAL_RECURSION_DEPTH;

const NUM_VERTICES: usize = 2_usize.pow(RECURSION_DEPTH as u32) - 1;
const NUM_INDICES: usize = (NUM_VERTICES - 1) * 2;

const BACKGROUND_COLOR: [f64; 3] = [0.0, 0.0, 0.0];

const WORK_GROUP_SIZE: usize = 2_usize.pow(WORK_GROUP_INITIAL_RECURSION_DEPTH as u32 - 1);

const MIN_BUFFER_SIZE: usize = 2056;

pub struct State {
    window: winit::window::Window,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    render_bind_group: wgpu::BindGroup,
    compute_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    direction_buffer: wgpu::Buffer,
    compute_uniform_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    render_uniform_buffer: wgpu::Buffer,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(Default::default());
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.describe().srgb)
            .cloned()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![], //TODO check color thingies
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_shader.wgsl"));

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                (NUM_VERTICES as u64 * Vertex::byte_size() as u64)
                                    .try_into()
                                    .expect("NUM_VERTICES should not be 0"),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                (NUM_VERTICES as u64 * Vertex::byte_size() as u64)
                                    .try_into()
                                    .expect("NUM_VERTICES should not be 0"),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                ..Default::default()
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            (std::mem::size_of::<[f32; 4]>() as u64).try_into().unwrap(),
                        ),
                    },
                    count: None,
                }],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                ..Default::default()
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VERTEY_BUFFER_LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let compute_offset = std::mem::size_of::<ComputeUniform>()
            .next_multiple_of(wgpu::Limits::default().min_uniform_buffer_offset_alignment as usize);

        let mut indices = [0u32; NUM_INDICES];
        // let compute_uniform = [0u8; ];

        for i in 0..NUM_INDICES / 2 {
            indices[2 * i] = (i / 2) as u32;
            indices[2 * i + 1] = (i + 1) as u32;
        }

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (NUM_VERTICES * Vertex::byte_size()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Uniform Buffer"),
            size: MIN_BUFFER_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: render_uniform_buffer.as_entire_binding(),
            }],
        });

        let direction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Direction Buffer"),
            size: (NUM_VERTICES * 2 * Vertex::byte_size()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Uniform Buffer"),
            size: MIN_BUFFER_SIZE.max(compute_offset * COMPUTE_RECURSION_DEPTH) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: direction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &compute_uniform_buffer,
                        offset: 0,
                        size: Some(
                            (std::mem::size_of::<ComputeUniform>() as u64)
                                .try_into()
                                .unwrap(),
                        ),
                    }),
                },
            ],
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            compute_pipeline,
            vertex_buffer,
            index_buffer,
            direction_buffer,
            compute_bind_group,
            render_bind_group,
            compute_uniform_buffer,
            render_uniform_buffer,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.size = size;
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(
        &mut self,
        mut hour: Vertex,
        mut minute: Vertex,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());

        let shrinking_factor = 0.6;

        hour.scale(shrinking_factor);
        minute.scale(shrinking_factor);

        let mut vertices = [Vertex { x: 0.0, y: 0.0 }; NUM_VERTICES]; //TODO unnessecary
        let mut directions = [Vertex { x: 0.0, y: 0.0 }; NUM_VERTICES * 2];
        directions[0] = hour;
        directions[1] = minute;

        directions[0].scale(0.6);
        directions[1].scale(0.6);

        for i in 0..WORK_GROUP_SIZE - 1 {
            vertices[i * 2 + 1] = vertices[i] + directions[i * 2];
            vertices[i * 2 + 2] = vertices[i] + directions[i * 2 + 1];

            directions[i * 4 + 2] = directions[i * 2] * hour;
            directions[i * 4 + 3] = directions[i * 2] * minute;
            directions[i * 4 + 4] = directions[i * 2 + 1] * hour;
            directions[i * 4 + 5] = directions[i * 2 + 1] * minute;
        }

        let color: [f32; 4] = [0.2, 1.0, 0.2, 0.25];

        self.queue
            .write_buffer(&self.render_uniform_buffer, 0, bytemuck::cast_slice(&color));

        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        self.queue
            .write_buffer(&self.direction_buffer, 0, bytemuck::cast_slice(&directions));

        let compute_offset = std::mem::size_of::<ComputeUniform>()
            .next_multiple_of(wgpu::Limits::default().min_uniform_buffer_offset_alignment as usize);

        // let mut compute_uniforms: Vec<u8> = Vec::with_capacity(
        //     compute_offset * (COMPUTE_RECURSION_DEPTH),
        // );

        let mut input_offset = (WORK_GROUP_SIZE - 1) as u32;
        let mut output_offset = input_offset * 2;

        let mut offset = 0;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute encoder"),
            });

        encoder.push_debug_group("Computing");
        for i in 0..(COMPUTE_RECURSION_DEPTH) as u32 {
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                });
                compute_pass.set_pipeline(&self.compute_pipeline);

                compute_pass.set_bind_group(0, &self.compute_bind_group, &[offset]);
                compute_pass.dispatch_workgroups(2_u32.pow(i), 1, 1);
            }

            self.queue.write_buffer(
                &self.compute_uniform_buffer,
                offset as u64,
                bytemuck::bytes_of(&ComputeUniform {
                    hour: hour,
                    minute: minute,
                    input_offset,
                    output_offset,
                }),
            );
            input_offset += WORK_GROUP_SIZE as u32 * 2_u32.pow(i);
            output_offset += WORK_GROUP_SIZE as u32 * 2_u32.pow(i + 1);

            offset += compute_offset as u32;
        }
        encoder.pop_debug_group();

        self.queue.submit(Some(encoder.finish()));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render encoder"),
            });

        encoder.push_debug_group("Rendering");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: BACKGROUND_COLOR[0],
                            g: BACKGROUND_COLOR[1],
                            b: BACKGROUND_COLOR[2],
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..NUM_INDICES as u32, 0, 0..1);
        }
        encoder.pop_debug_group();

        self.queue.submit(Some(encoder.finish()));

        output.present();

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    x: f32,
    y: f32,
}

impl Vertex {
    const fn byte_size() -> usize {
        std::mem::size_of::<Vertex>()
    }

    pub fn scale(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
    }

    pub fn len(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

impl From<(f32, f32)> for Vertex {
    fn from(value: (f32, f32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl Add for Vertex {
    type Output = Vertex;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul for Vertex {
    type Output = Vertex;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.x * rhs.y + self.y * rhs.x,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeUniform {
    hour: Vertex,
    minute: Vertex,
    input_offset: u32,
    output_offset: u32,
}

const VERTEY_BUFFER_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: Vertex::byte_size() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
};
