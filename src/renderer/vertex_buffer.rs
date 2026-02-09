use wgpu::util::DeviceExt;

use crate::renderer::{
    elements::{
        element_type::ElementType,
        owl::{OwlEdge, OwlNode, OwlType},
        rdfs::{RdfsEdge, RdfsType},
    },
    node_shape::NodeShape,
};

// Number of segments to divide each Bézier curve into for strip generation
const BEZIER_SEGMENTS: usize = 24;

// --- Uniforms ---
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniforms {
    pub resolution: [f32; 2],
    pub pan: [f32; 2],
    pub zoom: f32,
    /// WGSL struct padding.
    /// WebGL requires the uniform to be 16-byte aligned.
    pub _padding: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub quad_pos: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const VERTICES: &[Vertex] = &[
    Vertex {
        quad_pos: [-1.0, -1.0],
    },
    Vertex {
        quad_pos: [1.0, -1.0],
    },
    Vertex {
        quad_pos: [1.0, 1.0],
    },
    Vertex {
        quad_pos: [-1.0, -1.0],
    },
    Vertex {
        quad_pos: [1.0, 1.0],
    },
    Vertex {
        quad_pos: [-1.0, 1.0],
    },
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeInstance {
    pub position: [f32; 2],
    pub node_type: u32,
    pub shape_type: u32,
    pub shape_dim: [f32; 2],
    pub hovered: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MenuUniforms {
    pub center: [f32; 2],
    pub radius_inner: f32,
    pub radius_outer: f32,
    pub hovered_segment: i32,
    pub _padding: [u32; 7],
}

impl NodeInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![1 => Float32x2, 2 => Uint32, 3 => Uint32, 4 => Float32x2, 5 => Uint32];

    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub fn build_node_instances(
    positions: &[[f32; 2]],
    elements: &[ElementType],
    node_shapes: &[NodeShape],
    hovered_index: i32,
) -> Vec<NodeInstance> {
    let mut node_instances: Vec<NodeInstance> = vec![];
    for (i, pos) in positions.iter().enumerate() {
        let (shape_type, shape_dim) = match node_shapes[i] {
            NodeShape::Circle { r } => (0, [r, 0.0]),
            NodeShape::Rectangle { w, h } => (1, [w, h]),
        };
        #[expect(clippy::cast_possible_wrap)]
        let hovered = u32::from(i as i32 == hovered_index);
        node_instances.push(NodeInstance {
            position: *pos,
            node_type: elements[i].into(),
            shape_type,
            shape_dim,
            hovered,
        });
    }
    node_instances
}

pub fn create_node_instance_buffer(
    device: &wgpu::Device,
    positions: &[[f32; 2]],
    elements: &[ElementType],
    node_shapes: &[NodeShape],
    hovered_index: i32,
) -> wgpu::Buffer {
    let node_instances = build_node_instances(positions, elements, node_shapes, hovered_index);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance_node_buffer"),
        contents: bytemuck::cast_slice(&node_instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

// Edge vertex for Bézier strip rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeVertex {
    pub position: [f32; 2],       // Actual position in pixel space
    pub t_param: f32,             // Parameter t along curve [0..1]
    pub side: i32,                // -1 or +1 for left/right side of strip
    pub line_type: u32,           // Line style
    pub end_shape_type: u32,      // Shape at end node
    pub end_shape_dim: [f32; 2],  // Dimensions of end shape
    pub curve_start: [f32; 2],    // Curve start point (for arrow calculation)
    pub curve_end: [f32; 2],      // Curve end point (for arrow calculation)
    pub tangent_at_end: [f32; 2], // Tangent direction at t=1
    pub ctrl: [f32; 2],           // Control point for quadratic Bezier
    pub hovered: u32,
}

impl EdgeVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 11] = wgpu::vertex_attr_array![
        0 => Float32x2,  // position
        1 => Float32,    // t_param
        2 => Sint32,     // side
        3 => Uint32,     // line_type
        4 => Uint32,     // end_shape_type
        5 => Float32x2,  // end_shape_dim
        6 => Float32x2,  // curve_start
        7 => Float32x2,  // curve_end
        8 => Float32x2,  // tangent_at_end
        9 => Float32x2,  // ctrl
        10 => Uint32,    // hover
    ];

    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// Evaluate quadratic Bézier curve
fn bezier_point(p0: [f32; 2], ctrl: [f32; 2], p2: [f32; 2], t: f32) -> [f32; 2] {
    let t1 = 1.0 - t;
    [
        (t * t).mul_add(p2[0], (t1 * t1).mul_add(p0[0], 2.0 * t1 * t * ctrl[0])),
        (t * t).mul_add(p2[1], (t1 * t1).mul_add(p0[1], 2.0 * t1 * t * ctrl[1])),
    ]
}

// Evaluate derivative of quadratic Bézier curve
fn bezier_tangent(p0: [f32; 2], ctrl: [f32; 2], p2: [f32; 2], t: f32) -> [f32; 2] {
    // B'(t) = 2(1-t)(ctrl - p0) + 2t(p2 - ctrl)
    let t1 = 1.0 - t;
    [
        (2.0 * t1).mul_add(ctrl[0] - p0[0], 2.0 * t * (p2[0] - ctrl[0])),
        (2.0 * t1).mul_add(ctrl[1] - p0[1], 2.0 * t * (p2[1] - ctrl[1])),
    ]
}

fn normalize(v: [f32; 2]) -> [f32; 2] {
    let len: f32 = v[0].hypot(v[1]);
    if len > 1e-6 {
        [v[0] / len, v[1] / len]
    } else {
        [0.0, 1.0]
    }
}

pub fn build_line_and_arrow_vertices(
    edges: &[[usize; 3]],
    node_positions: &[[f32; 2]],
    node_shapes: &[NodeShape],
    elements: &[ElementType],
    zoom: f32,
    hovered_index: i32,
) -> (Vec<EdgeVertex>, Vec<EdgeVertex>) {
    const LINE_THICKNESS: f32 = 2.25;
    const ARROW_LENGTH_PX: f32 = 10.0;
    const ARROW_WIDTH_PX: f32 = 15.0;
    const SHADER_DIAMOND_LENGTH_PX: f32 = ARROW_LENGTH_PX * 2.0;
    const SHADER_DIAMOND_WIDTH_PX: f32 = ARROW_WIDTH_PX + 5.0;
    const ARROW_PADDING_PX: f32 = 5.0;

    let mut line_vertices: Vec<EdgeVertex> = Vec::new();
    let mut arrow_vertices: Vec<EdgeVertex> = Vec::new();
    let radius_pix = 50.0;

    for &[start_idx, center_idx, end_idx] in edges {
        let mut start = node_positions[start_idx];
        let center = node_positions[center_idx];
        let mut end = node_positions[end_idx];

        let line_type = match elements[center_idx] {
            ElementType::Rdfs(RdfsType::Edge(RdfsEdge::SubclassOf)) => 1,
            ElementType::Owl(OwlType::Edge(OwlEdge::DisjointWith)) => 2,
            ElementType::Owl(OwlType::Edge(OwlEdge::ValuesFrom)) => 3,
            _ => match elements[end_idx] {
                ElementType::Owl(OwlType::Node(node)) => match node {
                    OwlNode::UnionOf
                    | OwlNode::DisjointUnion
                    | OwlNode::Complement
                    | OwlNode::IntersectionOf => 4,
                    _ => {
                        if matches!(elements[center_idx], ElementType::NoDraw) {
                            continue;
                        }
                        0
                    }
                },
                _ => 0,
            },
        };

        let (end_shape_type, end_shape_dim) = match node_shapes[end_idx] {
            NodeShape::Circle { r } => (0, [r, 0.0]),
            NodeShape::Rectangle { w, h } => (1, [w, h]),
        };

        let mut start_shape = node_shapes[start_idx];
        let mut end_shape = node_shapes[end_idx];

        // Handle symmetric properties
        if start_idx == end_idx {
            let node_center = node_positions[start_idx];
            if let NodeShape::Circle { r } = node_shapes[start_idx] {
                let dx = center[0] - node_center[0];
                let dy = center[1] - node_center[1];
                let angle = f32::atan2(dy, dx);
                let offset_angle = angle + std::f32::consts::FRAC_PI_2;
                let offset_x = offset_angle.cos() * radius_pix * 0.5;
                let offset_y = offset_angle.sin() * radius_pix * 0.5;

                start = [
                    (r * radius_pix / 4.0).mul_add(angle.cos(), node_center[0] + offset_x),
                    (r * radius_pix / 4.0).mul_add(angle.sin(), node_center[1] + offset_y),
                ];
                end = [
                    (r * radius_pix / 4.0).mul_add(angle.cos(), node_center[0] - offset_x),
                    (r * radius_pix / 4.0).mul_add(angle.sin(), node_center[1] - offset_y),
                ];

                let half_shape = match start_shape {
                    NodeShape::Circle { r } => NodeShape::Circle { r: r / 2.0 },
                    NodeShape::Rectangle { w, h } => NodeShape::Rectangle { w: w / 2.0, h },
                };
                start_shape = half_shape;
                end_shape = half_shape;
            }
        }

        let start_center = start;
        let end_center = end;

        // Adjust start point to node perimeter
        let dir_start = [center[0] - start_center[0], center[1] - start_center[1]];
        let start_len = dir_start[0].hypot(dir_start[1]);
        let dir_start_n = if start_len > 1e-6 {
            [dir_start[0] / start_len, dir_start[1] / start_len]
        } else {
            [0.0, -1.0]
        };

        let perimeter_offset = 0.025;
        if start_idx != center_idx {
            start = match start_shape {
                NodeShape::Circle { r } => [
                    (dir_start_n[0] * (r - perimeter_offset)).mul_add(radius_pix, start_center[0]),
                    (dir_start_n[1] * (r - perimeter_offset)).mul_add(radius_pix, start_center[1]),
                ],
                NodeShape::Rectangle { w, h } => {
                    let dx = dir_start_n[0];
                    let dy = dir_start_n[1];
                    let mut scale = f32::INFINITY;
                    if dx.abs() > 1e-6 {
                        scale = scale.min((w * 0.9 / 2.0) / dx.abs());
                    }
                    if dy.abs() > 1e-6 {
                        scale = scale.min((h * 0.25 / 2.0) / dy.abs());
                    }
                    if scale.is_finite() {
                        [
                            (dir_start_n[0] * scale).mul_add(radius_pix, start_center[0]),
                            (dir_start_n[1] * scale).mul_add(radius_pix, start_center[1]),
                        ]
                    } else {
                        [start_center[0], start_center[1]]
                    }
                }
            };
        }

        // Adjust end point to node perimeter
        let dir_end = [center[0] - end_center[0], center[1] - end_center[1]];
        let end_len = dir_end[0].hypot(dir_end[1]);
        let dir_end_n = if end_len > 1e-6 {
            [dir_end[0] / end_len, dir_end[1] / end_len]
        } else {
            [0.0, 1.0]
        };

        if end_idx != center_idx {
            end = match end_shape {
                NodeShape::Circle { r } => [
                    (dir_end_n[0] * (r - perimeter_offset)).mul_add(radius_pix, end_center[0]),
                    (dir_end_n[1] * (r - perimeter_offset)).mul_add(radius_pix, end_center[1]),
                ],
                NodeShape::Rectangle { w, h } => {
                    let dx = dir_end_n[0];
                    let dy = dir_end_n[1];
                    let mut scale = f32::INFINITY;
                    if dx.abs() > 1e-6 {
                        scale = scale.min((w * 0.9) / dx.abs());
                    }
                    if dy.abs() > 1e-6 {
                        scale = scale.min((h * 0.25) / dy.abs());
                    }
                    if scale.is_finite() {
                        [
                            (dir_end_n[0] * scale).mul_add(radius_pix, end_center[0]),
                            (dir_end_n[1] * scale).mul_add(radius_pix, end_center[1]),
                        ]
                    } else {
                        [end_center[0], end_center[1]]
                    }
                }
            };
        }

        #[expect(clippy::cast_possible_wrap)]
        let hovered = u32::from(center_idx as i32 == hovered_index);

        // Compute control point for quadratic Bézier
        let ctrl = [
            (4.0f32.mul_add(center[0], -start[0]) - end[0]) * 0.5,
            (4.0f32.mul_add(center[1], -start[1]) - end[1]) * 0.5,
        ];

        // Calculate tangent at end for arrow
        let tangent_at_end = normalize(bezier_tangent(start, ctrl, end, 1.0));

        // Generate strip vertices
        let first_strip = line_vertices.is_empty();

        if let Some(prev_last) = line_vertices.last() {
            // duplicate previous last to avoid breaks
            line_vertices.push(*prev_last);
        }

        for i in 0..=BEZIER_SEGMENTS {
            let t = i as f32 / BEZIER_SEGMENTS as f32;
            let point = bezier_point(start, ctrl, end, t);
            let tangent = normalize(bezier_tangent(start, ctrl, end, t));

            // Perpendicular to tangent (left side)
            let perp = [-tangent[1], tangent[0]];

            // push left and right vertices for the strip
            let thickness = LINE_THICKNESS * zoom;
            let left = [
                perp[0].mul_add(thickness, point[0]),
                perp[1].mul_add(thickness, point[1]),
            ];
            let right = [
                perp[0].mul_add(-thickness, point[0]),
                perp[1].mul_add(-thickness, point[1]),
            ];

            if i == 0 && !first_strip {
                // duplicate first of new strip for degenerates
                line_vertices.push(EdgeVertex {
                    position: left,
                    t_param: t,
                    side: 1,
                    line_type,
                    end_shape_type,
                    end_shape_dim,
                    curve_start: start,
                    curve_end: end,
                    tangent_at_end,
                    ctrl,
                    hovered,
                });
            }

            line_vertices.push(EdgeVertex {
                position: left,
                t_param: t,
                side: 1,
                line_type,
                end_shape_type,
                end_shape_dim,
                curve_start: start,
                curve_end: end,
                tangent_at_end,
                ctrl,
                hovered,
            });
            line_vertices.push(EdgeVertex {
                position: right,
                t_param: t,
                side: -1,
                line_type,
                end_shape_type,
                end_shape_dim,
                curve_start: start,
                curve_end: end,
                tangent_at_end,
                ctrl,
                hovered,
            });
        }

        // Build arrow geometry at the tip
        let tip = end;
        let dir = normalize(tangent_at_end);
        let perp = [-dir[1], dir[0]];

        if line_type == 4 {
            // Diamond: generate two triangles
            let diamond_length = SHADER_DIAMOND_LENGTH_PX;
            let diamond_width = SHADER_DIAMOND_WIDTH_PX;

            let diamond_tip_padded = [
                dir[0].mul_add(ARROW_PADDING_PX, tip[0]),
                dir[1].mul_add(ARROW_PADDING_PX, tip[1]),
            ];
            let diamond_back_padded = [
                dir[0].mul_add(-(diamond_length + ARROW_PADDING_PX), tip[0]),
                dir[1].mul_add(-(diamond_length + ARROW_PADDING_PX), tip[1]),
            ];

            let diamond_center = [
                (dir[0] * diamond_length).mul_add(-0.5, tip[0]),
                (dir[1] * diamond_length).mul_add(-0.5, tip[1]),
            ];
            let halfw_padded = diamond_width.mul_add(0.5, ARROW_PADDING_PX);

            let diamond_left_padded = [
                perp[0].mul_add(halfw_padded, diamond_center[0]),
                perp[1].mul_add(halfw_padded, diamond_center[1]),
            ];
            let diamond_right_padded = [
                perp[0].mul_add(-halfw_padded, diamond_center[0]),
                perp[1].mul_add(-halfw_padded, diamond_center[1]),
            ];

            let common = |pos: [f32; 2]| EdgeVertex {
                position: pos,
                t_param: 1.0,
                side: 0,
                line_type,
                end_shape_type,
                end_shape_dim,
                curve_start: start,
                curve_end: end,
                tangent_at_end,
                ctrl,
                hovered,
            };

            // Triangle 1: tip, left, right
            arrow_vertices.push(common(diamond_tip_padded));
            arrow_vertices.push(common(diamond_left_padded));
            arrow_vertices.push(common(diamond_right_padded));

            // Triangle 2: back, right, left
            arrow_vertices.push(common(diamond_back_padded));
            arrow_vertices.push(common(diamond_right_padded));
            arrow_vertices.push(common(diamond_left_padded));
        } else {
            // Simple triangular arrowhead

            let tip_padded = [
                dir[0].mul_add(ARROW_PADDING_PX, tip[0]),
                dir[1].mul_add(ARROW_PADDING_PX, tip[1]),
            ];
            let base_center_padded = [
                dir[0].mul_add(-(ARROW_LENGTH_PX + ARROW_PADDING_PX), tip[0]),
                dir[1].mul_add(-(ARROW_LENGTH_PX + ARROW_PADDING_PX), tip[1]),
            ];
            let halfw_padded = ARROW_WIDTH_PX.mul_add(0.5, ARROW_PADDING_PX);
            let left_padded = [
                perp[0].mul_add(halfw_padded, base_center_padded[0]),
                perp[1].mul_add(halfw_padded, base_center_padded[1]),
            ];
            let right_padded = [
                perp[0].mul_add(-halfw_padded, base_center_padded[0]),
                perp[1].mul_add(-halfw_padded, base_center_padded[1]),
            ];

            let common = |pos: [f32; 2]| EdgeVertex {
                position: pos,
                t_param: 1.0,
                side: 0,
                line_type,
                end_shape_type,
                end_shape_dim,
                curve_start: start,
                curve_end: end,
                tangent_at_end,
                ctrl,
                hovered,
            };

            arrow_vertices.push(common(tip_padded));
            arrow_vertices.push(common(left_padded));
            arrow_vertices.push(common(right_padded));
        }
    }

    (line_vertices, arrow_vertices)
}

pub fn create_edge_vertex_buffer(
    device: &wgpu::Device,
    edges: &[[usize; 3]],
    node_positions: &[[f32; 2]],
    node_shapes: &[NodeShape],
    elements: &[ElementType],
    zoom: f32,
    hovered_index: i32,
) -> (wgpu::Buffer, u32, wgpu::Buffer, u32) {
    // Build separate vertex lists
    let (line_vertices, arrow_vertices) = build_line_and_arrow_vertices(
        edges,
        node_positions,
        node_shapes,
        elements,
        zoom,
        hovered_index,
    );

    let line_count = line_vertices.len() as u32;
    let arrow_count = arrow_vertices.len() as u32;

    let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge_line_vertex_buffer"),
        contents: bytemuck::cast_slice(&line_vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let arrow_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge_arrow_vertex_buffer"),
        contents: bytemuck::cast_slice(&arrow_vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    (line_buffer, line_count, arrow_buffer, arrow_count)
}
