//! Event channels for communicating with the renderer from the outside.

use crate::{graph_data::GraphDisplayData, prelude::ElementType};

/// Describes an event received by a render [`State`].
#[derive(PartialEq)]
pub enum RenderEvent {
    /// Pause graph simulation.
    Paused,

    /// Resume graph simulation
    Resumed,

    /// Zoom the graph.
    /// Negative values zoom out, positive zoom in.
    Zoomed(f64),

    /// Zoom to show all nodes
    CenterGraph,

    /// Loads a new graph from input
    LoadGraph(Box<GraphDisplayData>),
}
