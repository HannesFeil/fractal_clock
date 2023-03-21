#![feature(int_roundings)]
#![feature(array_chunks)]
#![cfg(target_pointer_width = "64")]

use std::{path::PathBuf, time::Duration};

use constants::{END_MILLIS, MILLIS_PER_FRAME, MINUTE_MILLIS, TOTAL_MILLIS};
use gui::{FractalClockRenderer, Vertex};
use ndarray::Array3;
use video_rs::{Encoder, EncoderSettings, Locator, Time};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use crate::constants::{BYTES_PER_PIXEL, RENDER_FORMAT, RENDER_SIZE};

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

    pub const RENDER_SIZE: u32 = 2000;

    pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub const BYTES_PER_PIXEL: u32 = 4;
    pub const UNPADDED_BYTES_PER_ROW: u32 = BYTES_PER_PIXEL * RENDER_SIZE;
    pub const BYTES_PER_ROW: u32 =
        UNPADDED_BYTES_PER_ROW.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    pub const BACKGROUND_COLOR: wgpu::Color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const SCALE: f32 = 0.25;
    pub const HOUR_SCALE: f32 = 0.5;
    pub const SHRINKING_FACTOR: f32 = 0.75;
    pub const TRANSPARENCY: f32 = 0.075;

    pub const MINUTE_MILLIS: u64 = 60 * 60 * 1000;
    pub const TOTAL_MILLIS: u64 = 12 * MINUTE_MILLIS;

    pub const START_MILLIS: u64 = 0;
    pub const END_MILLIS: u64 = TOTAL_MILLIS;
    pub const MILLIS_PER_FRAME: u64 = 100;
}

fn main() {
    assert_eq!(
        BYTES_PER_PIXEL,
        RENDER_FORMAT.describe().block_size as u32,
        "BYTES_PER_PIXEL has to match RENDER_FORMAT"
    );

    run();
}

pub fn run() {
    env_logger::init();
    video_rs::init().unwrap();

    let destination: Locator = PathBuf::from("rainbow.mp4").into();
    let settings =
        EncoderSettings::for_h264_yuv420p(RENDER_SIZE as usize, RENDER_SIZE as usize, false);

    let mut encoder = Encoder::new(&destination, settings).expect("failed to create encoder");

    // By determining the duration of each frame, we are essentially determing
    // the true frame rate of the output video. We choose 24 here.
    let duration: Time = Duration::from_millis(MILLIS_PER_FRAME).into();

    // Keep track of the current video timestamp.
    let mut position = Time::zero();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = FractalClockRenderer::new(window);

    let mut hour: Vertex = (1.0, 0.0).into();
    let mut minute: Vertex = (1.0, 0.0).into();

    let mut current_millis = constants::START_MILLIS;

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
            _ => {}
        },
        Event::MainEventsCleared => {
            let hour_angle =
                -2.0 * (current_millis as f32 / TOTAL_MILLIS as f32) * std::f32::consts::PI
                    + std::f32::consts::FRAC_PI_2;

            let minute_angle = -2.0
                * ((current_millis % MINUTE_MILLIS) as f32 / MINUTE_MILLIS as f32)
                * std::f32::consts::PI
                + std::f32::consts::FRAC_PI_2;

            println!(
                "current: {current_millis}, hour_angle: {hour_angle}, minute_angle: {minute_angle}"
            );

            current_millis += MILLIS_PER_FRAME;

            hour.scale(1.0 / hour.len());
            minute.scale(1.0 / minute.len());

            match state.render(
                hour_angle.sin_cos().into(),
                minute_angle.sin_cos().into(),
                [0.0, 1.0, 0.25],
                state.window().inner_size().height as f32
                    / state.window().inner_size().width as f32,
            ) {
                Ok(_) => {
                    let image = state.create_image();

                    assert_eq!(RENDER_SIZE as usize * RENDER_SIZE as usize * 3, image.len());

                    // This will create a smooth rainbow animation video!
                    let frame = Array3::from_shape_vec(
                        (RENDER_SIZE as usize, RENDER_SIZE as usize, 3),
                        state.create_image(),
                    )
                    .unwrap();

                    encoder
                        .encode(&frame, &position)
                        .expect("failed to encode frame");

                    // Update the current position and add `duration` to it.
                    position = position.aligned_with(&duration).add();

                    // gif_encoder
                    //     .encode_frame(image::Frame::from_parts(
                    //         state.create_image(),
                    //         0,
                    //         0,
                    //         image::Delay::from_saturating_duration(Duration::from_millis(
                    //             MILLIS_PER_FRAME,
                    //         )),
                    //     ))
                    //     .unwrap()
                }
                Err(wgpu::SurfaceError::Lost) => state.resize(state.window().inner_size()),
                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => eprintln!("{e:?}"),
            }

            if current_millis > END_MILLIS {
                control_flow.set_exit();
            }
        }
        Event::LoopDestroyed => {
            encoder.finish().expect("failed to finish encoder");
        }
        _ => {}
    });
}
