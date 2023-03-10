// Vertex shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

const vertices = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
);

@group(0) @binding(2)
var<uniform> aspect_ratio: f32;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos: vec2<f32>;

    switch in_vertex_index {
        case 0u: { pos = vertices[0]; }
        case 1u: { pos = vertices[3]; }
        case 2u: { pos = vertices[2]; }
        case 3u: { pos = vertices[1]; }
        case 4u: { pos = vertices[2]; }
        case 5u: { pos = vertices[3]; }
        default: { pos = vertices[0]; }
    }

    out.clip_position = vec4(pos.x * aspect_ratio, pos.y, 0.0, 1.0);
    out.tex_coords = saturate(vec2(pos.x, -pos.y));
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}