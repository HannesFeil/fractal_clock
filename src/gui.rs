use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages,
};
use winit::{dpi::PhysicalSize, window::Window};

const NUM_VERTICES: usize = 1 + 2_usize.pow(8);
const NUM_INDICES: usize = (NUM_VERTICES - 1) * 2;

const BACKGROUND_COLOR: [f64; 3] = [0.0, 0.0, 0.0];

const WORK_GROUP_SIZE: u32 = 64;
const WORK_GROUP_COUNT: u32 = 1;

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
    direction_uniform_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    color_uniform_buffer: wgpu::Buffer,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(Default::default());
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
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
                                (NUM_VERTICES as u64 * std::mem::size_of::<Vertex>() as u64)
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
                                (NUM_VERTICES as u64 * std::mem::size_of::<Vertex>() as u64)
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
                            has_dynamic_offset: false,
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
                    blend: Some(wgpu::BlendState::REPLACE),
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

        let mut vertices = [Vertex { position: [0.0; 2] }; NUM_VERTICES];
        let mut indices = [0; NUM_INDICES];
        let mut color = [0.0_f32; MIN_BUFFER_SIZE / std::mem::size_of::<f32>()];
        let mut direction =
            [Vertex { position: [0.0; 2] }; MIN_BUFFER_SIZE / std::mem::size_of::<Vertex>()];

        color[1] = 1.0;

        direction[0] = Vertex {
            position: [0.0, 1.0],
        };
        direction[1] = Vertex {
            position: [1.0, 0.0],
        };

        for i in 0..NUM_INDICES / 2 {
            indices[2 * i] = i / 2;
            indices[2 * i + 1] = i + 1;
        }

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE,
        });

        let color_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Color Uniform Buffer"),
            contents: bytemuck::cast_slice(&color),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_uniform_buffer.as_entire_binding(),
            }],
        });

        vertices[0] = Vertex { position: [0.0; 2] };

        let direction_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Direction Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::STORAGE,
        });

        let direction_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Direction Uniform Buffer"),
            contents: bytemuck::cast_slice(&direction),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::STORAGE,
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
                    resource: direction_uniform_buffer.as_entire_binding(),
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
            direction_uniform_buffer,
            color_uniform_buffer,
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

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(WORK_GROUP_COUNT, 1, 1);
        }

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
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..NUM_INDICES as u32, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));

        output.present();

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

const VERTEY_BUFFER_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
};
