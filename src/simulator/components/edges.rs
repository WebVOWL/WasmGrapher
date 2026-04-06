//! Components which make up an edge

use specs::{Component, Entity, VecStorage};

/// Vector of edges from one source to (potentially) multiple targets.
/// Source is implicitly the node from which this component is accessed.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Connects {
    pub targets: Vec<Entity>,
}
