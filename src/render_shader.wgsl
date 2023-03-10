// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct RenderUniform {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> render: RenderUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.y, model.position.x, 0.0, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return render.color;
}