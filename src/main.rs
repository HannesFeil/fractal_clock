#![feature(array_chunks)]
#![feature(int_roundings)]
#![cfg(target_pointer_width = "64")]

use std::{
    fs::{self, File},
    time::Instant,
};

use clap::command;
use constants::{HOUR_MILLIS, TOTAL_MILLIS};
use gui::{FractalClockRenderer, Vertex};
use image::GrayImage;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow::Poll, EventLoop},
    window::WindowBuilder,
};

use crate::constants::{BYTES_PER_PIXEL, RENDER_FORMAT};

mod gui;
mod constants {
    /// Initial recursion depth, depending on work group size
    pub const WORK_GROUP_INITIAL_RECURSION_DEPTH: usize = 9;
    pub const WORK_GROUP_SIZE: usize = 2_usize.pow(WORK_GROUP_INITIAL_RECURSION_DEPTH as u32 - 1);

    /// Minimum wgpu uniform buffer size
    pub const MIN_BUFFER_SIZE: usize = 2056;

    pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub const BYTES_PER_PIXEL: u32 = 4;

    pub const BACKGROUND_COLOR: wgpu::Color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const SCALE: f32 = 0.25;
    pub const HOUR_SCALE: f32 = 0.8;
    pub const SHRINKING_FACTOR: f32 = 0.75;
    pub const TRANSPARENCY: f32 = 0.075;

    pub const HOUR_MILLIS: u64 = 60 * 60 * 1000;
    pub const TOTAL_MILLIS: u64 = 12 * HOUR_MILLIS;
}

#[derive(clap::Parser)]
#[command(name = "Fractal Clock")]
#[command(author, version, about = "Renders a fractal clock", long_about = None)]
struct Args {
    /// The width of the video being rendered.
    width: u32,

    /// The height of the video being rendered.
    height: u32,

    /// The total recurion depth
    #[arg(short, long, default_value_t = 16)]
    recursion_depth: usize,

    /// The starting time.
    #[arg(short, long, default_value_t = 0)]
    start_millis: u64,

    /// The ending time.
    #[arg(short, long, default_value_t = TOTAL_MILLIS)]
    end_millis: u64,

    /// How many millis time per frame rendered
    #[arg(long, default_value_t = 100)]
    millis_per_frame: u64,
}

fn main() {
    assert_eq!(
        BYTES_PER_PIXEL,
        RENDER_FORMAT.block_size(None).unwrap(),
        "BYTES_PER_PIXEL has to match RENDER_FORMAT"
    );

    let args = <Args as clap::Parser>::parse();

    run(args);
}

fn run(args: Args) {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(Poll);
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut image = GrayImage::new(args.width, args.height);

    let mut state =
        FractalClockRenderer::new(window, (args.width, args.height), args.recursion_depth);

    let mut hour: Vertex = (1.0, 0.0).into();
    let mut minute: Vertex = (1.0, 0.0).into();

    let mut current_millis = args.start_millis;

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(size) => state.resize(*size),
                WindowEvent::RedrawRequested => {
                    let hour_angle =
                        -2.0 * (current_millis as f32 / TOTAL_MILLIS as f32) * std::f32::consts::PI
                            + std::f32::consts::FRAC_PI_2;

                    let minute_angle = -2.0
                        * ((current_millis % HOUR_MILLIS) as f32 / HOUR_MILLIS as f32)
                        * std::f32::consts::PI
                        + std::f32::consts::FRAC_PI_2;

                    // println!("Rendering frame: current millis = {current_millis}");

                    hour.scale(1.0 / hour.len());
                    minute.scale(1.0 / minute.len());

                    let _now = Instant::now();
                    match state.render(
                        hour_angle.sin_cos().into(),
                        minute_angle.sin_cos().into(),
                        [1.0, 1.0, 1.0],
                    ) {
                        Ok(_) => {
                            state.create_image(&mut image);

                            let hour = current_millis / HOUR_MILLIS;
                            let current_in_hour = current_millis % HOUR_MILLIS;

                            image::save_buffer(
                                format!("output/{hour}/fractal_clock_frame_{current_in_hour:07}.png"),
                                &image,
                                image.width(),
                                image.height(),
                                image::ColorType::L8,
                            )
                            .unwrap();
                        }
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window().inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("{e:?}"),
                    }

                    current_millis += args.millis_per_frame;

                    if current_millis > args.end_millis {
                        elwt.exit();
                    }

                    state.window().request_redraw();
                }
                _ => {}
            },
            _ => {}
        })
        .unwrap();
}
