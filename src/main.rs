#![feature(int_roundings)]
#![feature(array_chunks)]
#![cfg(target_pointer_width = "64")]

use gui::{State, Vertex};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod gui;

mod constants {
    /// Total recursion depth
    pub const RECURSION_DEPTH: usize = 23;

    /// Initial recursion depth, depending on work group size
    pub const WORK_GROUP_INITIAL_RECURSION_DEPTH: usize = 9;
    pub const WORK_GROUP_SIZE: usize = 2_usize.pow(WORK_GROUP_INITIAL_RECURSION_DEPTH as u32 - 1);

    pub const COMPUTE_RECURSION_DEPTH: usize = RECURSION_DEPTH - WORK_GROUP_INITIAL_RECURSION_DEPTH;

    pub const NUM_VERTICES: usize = 2_usize.pow(RECURSION_DEPTH as u32) - 1;
    pub const NUM_INDICES: usize = (NUM_VERTICES - 1) * 2;

    /// Minimum wgpu uniform buffer size
    pub const MIN_BUFFER_SIZE: usize = 2056;

    pub const WIDTH: u32 = 2000;
    pub const HEIGHT: u32 = 2000;

    pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    pub const BYTES_PER_PIXEL: u32 = 4;
    pub const UNPADDED_BYTES_PER_ROW: u32 = BYTES_PER_PIXEL * WIDTH;
    pub const BYTES_PER_ROW: u32 =
        UNPADDED_BYTES_PER_ROW.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    pub const BACKGROUND_COLOR: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    /// TODO make variable instead of constants?
    pub const SCALE: f32 = 0.25;
    pub const HOUR_SCALE: f32 = 0.5;
    pub const SHRINKING_FACTOR: f32 = 0.75;
    pub const TRANSPARENCY: f32 = 0.005;
}

fn main() {
    run();
}

pub fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window);

    let mut hour: Vertex = (1.0, 0.0).into();
    let mut minute: Vertex = (1.0, 0.0).into();

    let mut cursor_buttons = (false, false);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => match event {
            WindowEvent::CloseRequested => control_flow.set_exit(),
            WindowEvent::Resized(size) => state.resize(*size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size)
            }
            WindowEvent::CursorMoved { position, .. } => {
                let normalized_pos = (
                    -(position.y as f32 / state.window().inner_size().height as f32 - 0.5),
                    position.x as f32 / state.window().inner_size().width as f32 - 0.5,
                );

                if cursor_buttons.0 {
                    hour = normalized_pos.into();
                }
                if cursor_buttons.1 {
                    minute = normalized_pos.into();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = matches!(state, ElementState::Pressed);
                match button {
                    MouseButton::Left => cursor_buttons.0 = pressed,
                    MouseButton::Right => cursor_buttons.1 = pressed,
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        state: winit::event::ElementState::Pressed,
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Space),
                        ..
                    },
                ..
            } => {
                state.print_image();
            }
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            hour.scale(1.0 / hour.len());
            minute.scale(1.0 / minute.len());

            match state.render(
                hour,
                minute,
                constants::WIDTH as f32 / constants::HEIGHT as f32,
            ) {
                Ok(()) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.window().inner_size()),
                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => eprintln!("{e:?}"),
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}
