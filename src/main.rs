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
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            hour.scale(1.0 / hour.len());
            minute.scale(1.0 / minute.len());

            match state.render(
                hour,
                minute,
                state.window().inner_size().height as f32
                    / state.window().inner_size().width as f32,
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
