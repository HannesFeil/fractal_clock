use std::ops::{Add, Mul};

use wgpu::BufferUsages;
use winit::{dpi::PhysicalSize, window::Window};

use crate::constants::*;

/// Containing wgpu related basic stuff
struct WGPUState {
    /// The window
    window: Window,
    /// Handle to the `Surface` of the window
    surface: wgpu::Surface,
    /// Handle to the `Device`
    device: wgpu::Device,
    /// Handle to the devices queue
    queue: wgpu::Queue,
    /// surface configuration
    config: wgpu::SurfaceConfiguration,
}

impl WGPUState {
    /// Creates a new `WGPUState` with the given `window`
    fn new(window: Window) -> Self {
        let instance = wgpu::Instance::new(Default::default());
        let surface =
            unsafe { instance.create_surface(&window) }.expect("Should not run on WebGL2");

        // Request any adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Expected a compatible adapter");

        // request device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                limits: wgpu::Limits::default(),
                label: Some("Device"),
            },
            None,
        ))
        .expect("Expected device supporting specified features");

        // Set up surface configuration
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = *surface_caps
            .formats
            .first()
            .expect("Expected preferred format");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Self {
            window,
            surface,
            device,
            queue,
            config,
        }
    }
}

/// Container for all used buffers
struct Buffers {
    /// Vertex buffer
    vertex_buffer: wgpu::Buffer,
    /// Buffer used to compute vertices
    direction_buffer: wgpu::Buffer,
    /// Buffer containing information for each compute call
    compute_uniform_buffer: wgpu::Buffer,
    /// Index buffer
    index_buffer: wgpu::Buffer,
    /// Buffer containing information for each rendering call
    render_uniform_buffer: wgpu::Buffer,
    /// Buffer containing aspect ratio for rendering to window
    window_render_uniform_buffer: wgpu::Buffer,
    /// Target buffer for copying the `render_texture`
    render_output_buffer: wgpu::Buffer,
}

impl Buffers {
    /// Creates the buffers with the given device
    fn new(
        device: &wgpu::Device,
        render_size: (u64, u64),
        compute_rec_depth: usize,
        num_vertices: usize,
    ) -> Self {
        let direction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Direction Buffer"),
            size: (num_vertices * 2 * Vertex::byte_size()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (num_vertices * Vertex::byte_size()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // dynamic offset steps for the compute uniform buffer
        let compute_offset = std::mem::size_of::<ComputeUniform>()
            .next_multiple_of(wgpu::Limits::default().min_uniform_buffer_offset_alignment as usize);

        let compute_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Uniform Buffer"),
            size: MIN_BUFFER_SIZE.max(compute_offset * compute_rec_depth) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            usage: BufferUsages::INDEX,
            size: ((num_vertices - 1) * 2 * std::mem::size_of::<u32>()) as u64,
            mapped_at_creation: true,
        });

        const CHUNK_SIZE: usize = std::mem::size_of::<u32>();

        let mut idx1 = 0_u32;
        let mut idx2 = 0_u32;
        let mut par = true;

        // Calculate indices: [0, 1, 0, 2, 1, 3, 1, 4, 2, 5, 2, 6, ...]
        for chunk in index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .array_chunks_mut::<CHUNK_SIZE>()
        {
            let val = if par {
                idx1
            } else {
                idx2 += 1;

                if idx2 % 2 == 0 {
                    idx1 += 1;
                }

                idx2
            };

            par = !par;

            *chunk = val.to_le_bytes();
        }
        index_buffer.unmap();

        let render_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Uniform Buffer"),
            size: MIN_BUFFER_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let window_render_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Window Render Uniform Buffer"),
            size: MIN_BUFFER_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Output Buffer"),
            size: (render_size.0 as u32 * BYTES_PER_PIXEL)
                .next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64
                * render_size.1,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            direction_buffer,
            compute_uniform_buffer,
            index_buffer,
            render_uniform_buffer,
            window_render_uniform_buffer,
            render_output_buffer,
        }
    }
}

/// Container for the rendering pipeline and related stuff
///
/// There are 3 stages in the rendering pipeline:
/// 1.  Computing the vertices of the fractal
/// 2.  Drawing lines between the vertices onto the `render_texture`
/// 3.  Sampling the texture and drawing it to the window
pub struct FractalClockRenderer {
    /// width and height of the rendered clock in pixels
    render_size: (u32, u32),
    /// Total recursion depth
    recursion_depth: usize,
    /// window, device, queue etc.
    state: WGPUState,
    /// Rendering pipeline
    render_pipeline: wgpu::RenderPipeline,
    /// Rendering bindgroup
    render_bind_group: wgpu::BindGroup,
    /// Computing pipeline
    compute_pipeline: wgpu::ComputePipeline,
    /// Computing bindgroup
    compute_bind_group: wgpu::BindGroup,
    /// Window rendering pipeline
    window_render_pipeline: wgpu::RenderPipeline,
    /// Window rendering bindgroup
    window_render_bind_group: wgpu::BindGroup,
    /// Render target texture
    render_texture: wgpu::Texture,
    /// Various [wgpu::Buffer]s
    buffers: Buffers,
}

impl FractalClockRenderer {
    /// Initializes a new FractalClockRenderer for the given window
    pub fn new(window: Window, render_size: (u32, u32), recursion_depth: usize) -> Self {
        let state = WGPUState::new(window);

        // Set up compute pipeline
        let compute_shader = state
            .device
            .create_shader_module(wgpu::include_wgsl!("compute_shader.wgsl"));

        let compute_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind Group Layout"),
                    entries: &[
                        // Vertices buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(
                                    ((2_u64.pow(recursion_depth as u32) - 1)
                                        * Vertex::byte_size() as u64)
                                        .try_into()
                                        .expect("NUM_VERTICES should not be 0"),
                                ),
                            },
                            count: None,
                        },
                        // Direction buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(
                                    ((2_u64.pow(recursion_depth as u32) - 1)
                                        * Vertex::byte_size() as u64)
                                        .try_into()
                                        .expect("NUM_VERTICES should not be 0"),
                                ),
                            },
                            count: None,
                        },
                        // Compute uniform buffer
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
            state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    ..Default::default()
                });

        let compute_pipeline =
            state
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &compute_shader,
                    entry_point: "main",
                });

        // Set up render pipeline
        let render_shader = state
            .device
            .create_shader_module(wgpu::include_wgsl!("render_shader.wgsl"));

        let render_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Render Bind Group Layout"),
                    entries: &[
                        // Render uniform buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let render_pipeline_layout =
            state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&render_bind_group_layout],
                    ..Default::default()
                });

        let render_pipeline =
            state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &render_shader,
                        entry_point: "vs_main",
                        buffers: &[VERTEY_BUFFER_LAYOUT],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &render_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                    format: RENDER_FORMAT,
                    blend:
                    // Custom blending, this works well?
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::DstAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        // Windowing pipeline
        let window_render_shader = state
            .device
            .create_shader_module(wgpu::include_wgsl!("window_render_shader.wgsl"));

        let window_render_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Window Render Bind Group Layout"),
                    entries: &[
                        // Render texture
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
                        // Render texture sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let window_render_pipeline_layout =
            state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Window Render Pipeline"),
                    bind_group_layouts: &[&window_render_bind_group_layout],
                    ..Default::default()
                });

        let window_render_pipeline =
            state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Window Render Pipeline"),
                    layout: Some(&window_render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &window_render_shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
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
                    fragment: Some(wgpu::FragmentState {
                        module: &window_render_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: state.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::all(),
                        })],
                    }),
                    multiview: None,
                });

        let buffers = Buffers::new(
            &state.device,
            (render_size.0 as u64, render_size.1 as u64),
            recursion_depth - WORK_GROUP_INITIAL_RECURSION_DEPTH,
            2_usize.pow(recursion_depth as u32) - 1,
        );

        let render_texture = state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: render_size.0,
                height: render_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RENDER_FORMAT,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let render_texture_sampler = state.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Render Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let render_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.render_uniform_buffer.as_entire_binding(),
            }],
        });

        let compute_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.direction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers.compute_uniform_buffer,
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

        let window_render_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Window Render Bind Group"),
            layout: &window_render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &render_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&render_texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.window_render_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            render_size,
            recursion_depth,
            state,
            render_pipeline,
            compute_pipeline,
            window_render_pipeline,
            render_bind_group,
            compute_bind_group,
            window_render_bind_group,
            render_texture,
            buffers,
        }
    }

    /// Return a reference to the window
    pub fn window(&self) -> &Window {
        &self.state.window
    }

    /// Update surface config with new size
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.state.config.width = size.width;
            self.state.config.height = size.height;
            self.state
                .surface
                .configure(&self.state.device, &self.state.config);
        }
    }

    /// Render the clock to the texture and surface and return the [wgpu::SubmissionIndex].
    pub fn render(
        &mut self,
        mut hour: Vertex,
        mut minute: Vertex,
        [r, g, b]: [f32; 3],
    ) -> Result<wgpu::SubmissionIndex, wgpu::SurfaceError> {
        let surface_output = self.state.surface.get_current_texture()?;
        let render_view = self
            .render_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("Render View"),
                aspect: wgpu::TextureAspect::All,
                ..Default::default()
            });
        let surface_view = surface_output.texture.create_view(&Default::default());

        // Calculate vertices up to WORK_GROUP_INITIAL_RECURSION_DEPTH
        let mut vertices = [Vertex { x: 0.0, y: 0.0 }; 2 * WORK_GROUP_SIZE + 1];
        let mut directions = [Vertex { x: 0.0, y: 0.0 }; 4 * WORK_GROUP_SIZE + 1];
        directions[0] = hour;
        directions[1] = minute;

        directions[0].scale(SCALE * HOUR_SCALE);
        directions[1].scale(SCALE);

        hour.scale(SHRINKING_FACTOR);
        minute.scale(SHRINKING_FACTOR);

        for i in 0..WORK_GROUP_SIZE - 1 {
            vertices[i * 2 + 1] = vertices[i] + directions[i * 2];
            vertices[i * 2 + 2] = vertices[i] + directions[i * 2 + 1];

            directions[i * 4 + 2] = directions[i * 2] * hour;
            directions[i * 4 + 3] = directions[i * 2] * minute;
            directions[i * 4 + 4] = directions[i * 2 + 1] * hour;
            directions[i * 4 + 5] = directions[i * 2 + 1] * minute;
        }

        // Set Render uniforms (color and transparency)
        let render_uniform: [f32; 5] = [
            r,
            g,
            b,
            TRANSPARENCY,
            self.render_size.0 as f32 / self.render_size.1 as f32,
        ];

        self.state.queue.write_buffer(
            &self.buffers.render_uniform_buffer,
            0,
            bytemuck::cast_slice(&render_uniform),
        );

        self.state.queue.write_buffer(
            &self.buffers.window_render_uniform_buffer,
            0,
            bytemuck::cast_slice(&[1.0_f32]),
        );

        self.state.queue.write_buffer(
            &self.buffers.vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );
        self.state.queue.write_buffer(
            &self.buffers.direction_buffer,
            0,
            bytemuck::cast_slice(&directions),
        );

        let compute_offset = std::mem::size_of::<ComputeUniform>()
            .next_multiple_of(wgpu::Limits::default().min_uniform_buffer_offset_alignment as usize);

        let mut input_offset = (WORK_GROUP_SIZE - 1) as u32;
        let mut output_offset = input_offset * 2;

        let mut offset = 0;

        let mut encoder =
            self.state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute encoder"),
                });

        // Calculate vertices, recursion layer by layer
        encoder.push_debug_group("Computing");
        for i in 0..(self.recursion_depth - WORK_GROUP_INITIAL_RECURSION_DEPTH) as u32 {
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                });
                compute_pass.set_pipeline(&self.compute_pipeline);

                compute_pass.set_bind_group(0, &self.compute_bind_group, &[offset]);
                compute_pass.dispatch_workgroups(2_u32.pow(i), 1, 1);
            }

            self.state.queue.write_buffer(
                &self.buffers.compute_uniform_buffer,
                offset as u64,
                bytemuck::bytes_of(&ComputeUniform {
                    hour,
                    minute,
                    input_offset,
                    output_offset,
                }),
            );
            input_offset += WORK_GROUP_SIZE as u32 * 2_u32.pow(i);
            output_offset += WORK_GROUP_SIZE as u32 * 2_u32.pow(i + 1);

            offset += compute_offset as u32;
        }
        encoder.pop_debug_group();

        // Render vertices
        encoder.push_debug_group("Rendering");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND_COLOR),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.buffers.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(
                0..((2_u32.pow(self.recursion_depth as u32) - 2) * 2),
                0,
                0..1,
            );
        }
        encoder.pop_debug_group();

        self.state.queue.submit(Some(encoder.finish()));

        encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute encoder"),
            });

        // Render to surface
        encoder.push_debug_group("Window Render Group");
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND_COLOR),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.window_render_pipeline);
            render_pass.set_bind_group(0, &self.window_render_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        encoder.pop_debug_group();

        // Store texture in buffer
        encoder.copy_texture_to_buffer(
            self.render_texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &self.buffers.render_output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        (self.render_size.0 * BYTES_PER_PIXEL)
                            .next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    ),
                    rows_per_image: Some(self.render_size.1),
                },
            },
            self.render_texture.size(),
        );

        let index = self.state.queue.submit(Some(encoder.finish()));
        surface_output.present();

        Ok(index)
    }

    /// Reads from the render_output_buffer after waiting for the most recent submission and returns a [image::RgbImage] created from it.
    pub fn create_image(&self, render_size: (u32, u32)) -> Vec<u8> {
        let output_slice = self.buffers.render_output_buffer.slice(..);

        let (notifyer, waiter) = oneshot::channel();

        output_slice.map_async(wgpu::MapMode::Read, move |r| notifyer.send(r).unwrap());

        self.state.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = waiter.recv() {
            let mapped = output_slice.get_mapped_range();

            let unpadded = mapped
                .chunks(
                    (render_size.0 * BYTES_PER_PIXEL)
                        .next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                        as usize,
                )
                .flat_map(|chunk| {
                    chunk[..(render_size.0 * BYTES_PER_PIXEL) as usize]
                        .chunks(BYTES_PER_PIXEL as usize)
                        .flat_map(|c| c.iter().take(3))
                })
                .cloned()
                .collect::<Vec<_>>();

            drop(mapped);
            self.buffers.render_output_buffer.unmap();

            unpadded
        } else {
            panic!("Failed async map on render_output buffer");
        }
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
