#![allow(dead_code)]
#![allow(unused)]

mod app;
mod events;
mod graph_data;
mod quadtree;
mod renderer;
mod simulator;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub use app::init_render;

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use app::run;

/// Exports all the core types of the library.
pub mod prelude {
    use crate::events::EventDispatcher;
    use std::sync::LazyLock;

    pub use crate::events::{
        gui_events::GUIEvent, render_event::RenderEvent, simulator_event::SimulatorEvent,
    };
    pub use crate::graph_data::{GraphDisplayData, GraphMetadata};
    pub use crate::quadtree::{BoundingBox2D, Node, QuadTree};
    pub use crate::renderer::elements::{
        characteristic::Characteristic, element_type::ElementType, generic::*, owl::*, rdf::*,
        rdfs::*, xsd::*,
    };
    pub use crate::simulator::Simulator;
    pub use crate::simulator::ressources::simulator_vars::{
        Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta, RepelForce,
        SpringNeutralLength, SpringStiffness,
    };

    // Re-export strum
    pub use strum;

    /// The global event handler
    pub static EVENT_DISPATCHER: LazyLock<EventDispatcher> = LazyLock::new(EventDispatcher::new);
}
