@group(0) @binding(0)
var<storage, read_write> vertices: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read_write> directions: array<vec2<f32>>;

struct TimeInput {
    hour: vec2<f32>,
    minute: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> time: TimeInput;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    vertices[id.x + 1u] = vertices[id.x];
    vertices[id.x + 1u].x += vertices[id.x].x;
    vertices[id.x + 1u].y += vertices[id.x].y;
}