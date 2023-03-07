@group(0) @binding(0)
var<storage, read_write> vertices: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> directions: array<vec4<f32>>;

struct Input {
    hour: vec2<f32>,
    minute: vec2<f32>,
    in_offset: u32,
    out_offset: u32,
}

@group(0) @binding(2)
var<uniform> input: Input;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let parent = vertices[index + input.in_offset];
    let dirs = directions[index + input.in_offset];

    let new_index = index * 2u + 1u + input.out_offset;

    vertices[new_index] = parent + dirs.xy;
    vertices[new_index + 1u] = parent + dirs.zw;
    directions[new_index] = vec4(multiply(dirs.xy, input.hour), multiply(dirs.xy, input.minute));
    directions[new_index + 1u] = vec4(multiply(dirs.zw, input.hour), multiply(dirs.zw, input.minute));
}

fn multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}