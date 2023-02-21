@group(0) @binding(0)
var<storage, read_write> vertices: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> directions: array<vec4<f32>>;

struct Input {
    hour: vec2<f32>,
    minute: vec2<f32>,
    in_offest: u32,
    out_offset: u32,
}

@group(0) @binding(0)
var<uniform> input: Input;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let parent = vertices[index];
    vertices[index + 1u] = vertices[index];
    vertices[index + 1u].x += 0.1;
    vertices[index + 1u].y += 0.1;
}