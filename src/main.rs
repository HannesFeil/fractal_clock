#![feature(int_roundings)]
#![feature(array_chunks)]
#![cfg(target_pointer_width = "64")]

use std::{error::Error, fmt::Display, path::PathBuf, str::FromStr, time::Duration};

use clap::{command, Parser};
use constants::{MINUTE_MILLIS, TOTAL_MILLIS};
use gui::{FractalClockRenderer, Vertex};
use ndarray::Array3;
use palette::{FromColor, Hsv, Srgb};
use video_rs::{Encoder, EncoderSettings, Locator, Time};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
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
    pub const HOUR_SCALE: f32 = 0.5;
    pub const SHRINKING_FACTOR: f32 = 0.75;
    pub const TRANSPARENCY: f32 = 0.075;

    pub const MINUTE_MILLIS: u64 = 60 * 60 * 1000;
    pub const TOTAL_MILLIS: u64 = 12 * MINUTE_MILLIS;
}

#[derive(clap::Parser)]
#[command(name = "Fractal Clock")]
#[command(author, version, about = "Renders a fractal clock", long_about = None)]
struct Args {
    /// The file which the rendered video will be written to.
    file: PathBuf,
    /// The width and height of the video being rendered.
    size: u32,
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
    /// Optionally set the color mode.
    #[arg(short, long, default_value_t = ColorMode::HSVDay)]
    color_mode: ColorMode,
}

#[derive(Debug)]
pub struct ColorModeError(String);

impl Display for ColorModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{invalid} is not a valid color mode. Valid modes are: {opts:?}",
            invalid = self.0,
            opts = ColorMode::VALID_OPTS,
        ))
    }
}

impl Error for ColorModeError {}

#[derive(Clone, Debug)]
pub enum ColorMode {
    HSVDay,
    Constant(f32, f32, f32),
}

impl Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorMode::HSVDay => f.write_str("hsvday"),
            ColorMode::Constant(r, g, b) => f.write_fmt(format_args!("constant({r},{g},{b})")),
        }
    }
}

impl ColorMode {
    const VALID_OPTS: &[&'static str] = &["constant(r[0.0-1.0],g[0.0-1.0],b[0.0-1.0])", "hsvday"];

    pub fn color(&self, time_millis: u64) -> [f32; 3] {
        match *self {
            ColorMode::Constant(r, g, b) => [r, g, b],
            ColorMode::HSVDay => {
                let (r, g, b) = Srgb::from_color(Hsv::new(
                    time_millis as f32 / TOTAL_MILLIS as f32 * 360.0,
                    1.0,
                    1.0,
                ))
                .into_linear()
                .into_components();

                [r, g, b]
            }
        }
    }
}

impl FromStr for ColorMode {
    type Err = ColorModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s == "hsvday" {
            Ok(ColorMode::HSVDay)
        } else if s.starts_with("constant") {
            let trimmed = s.trim_start_matches("constant");
            let vals = trimmed[1..trimmed.len() - 1]
                .split(',')
                .map(|v| v.parse::<f32>())
                .collect::<Vec<_>>();
            if let [Ok(r), Ok(g), Ok(b)] = vals.as_slice() {
                Ok(ColorMode::Constant(*r, *g, *b))
            } else {
                Err(ColorModeError(s.to_string()))
            }
        } else {
            Err(ColorModeError(s.to_string()))
        }
    }
}

fn main() {
    assert_eq!(
        BYTES_PER_PIXEL,
        RENDER_FORMAT.describe().block_size as u32,
        "BYTES_PER_PIXEL has to match RENDER_FORMAT"
    );

    let args = Args::parse();

    run(args);
}

fn run(args: Args) {
    env_logger::init();
    video_rs::init().unwrap();

    let destination: Locator = args.file.into();
    let settings = EncoderSettings::for_h264_yuv420p(args.size as usize, args.size as usize, false);

    let mut encoder = Encoder::new(&destination, settings).expect("failed to create encoder");

    // By determining the duration of each frame, we are essentially determing
    // the true frame rate of the output video. We choose 24 here.
    let duration: Time = Duration::from_millis(args.millis_per_frame).into();

    // Keep track of the current video timestamp.
    let mut position = Time::zero();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = FractalClockRenderer::new(window, args.size, args.recursion_depth);

    let mut hour: Vertex = (1.0, 0.0).into();
    let mut minute: Vertex = (1.0, 0.0).into();

    let mut current_millis = args.start_millis;

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

            println!("Rendering frame: current millis = {current_millis}");

            hour.scale(1.0 / hour.len());
            minute.scale(1.0 / minute.len());

            match state.render(
                hour_angle.sin_cos().into(),
                minute_angle.sin_cos().into(),
                args.color_mode.color(current_millis),
                state.window().inner_size().height as f32
                    / state.window().inner_size().width as f32,
            ) {
                Ok(_) => {
                    let frame = Array3::from_shape_vec(
                        (args.size as usize, args.size as usize, 3),
                        state.create_image(args.size),
                    )
                    .unwrap();

                    encoder
                        .encode(&frame, &position)
                        .expect("failed to encode frame");

                    // Update the current position and add `duration` to it.
                    position = position.aligned_with(&duration).add();
                }
                Err(wgpu::SurfaceError::Lost) => state.resize(state.window().inner_size()),
                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => eprintln!("{e:?}"),
            }

            current_millis += args.millis_per_frame;

            if current_millis > args.end_millis {
                control_flow.set_exit();
            }
        }
        Event::LoopDestroyed => {
            encoder.finish().expect("failed to finish encoder");
        }
        _ => {}
    });
}
