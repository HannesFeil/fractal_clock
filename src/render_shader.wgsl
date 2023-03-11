// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
    @builtin(vertex_index) index: u32,
}

struct RenderUniform {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> render: RenderUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) base_pointer: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.y, model.position.x, 0.0, 1.0);
    if model.index == 0u {
        out.base_pointer = 1.0;
    } else {
        out.base_pointer = 0.0;
    }
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.base_pointer > 0.0 {
        return vec4(1.0, 1.0, 1.0, 1.0);
    }

    return render.color;
}