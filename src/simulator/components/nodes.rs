//! Components which make up a node

use crate::renderer::elements::element_type::ElementType;
use glam::Vec2;
use specs::{Component, NullStorage, VecStorage};

/// The position of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Position(pub Vec2);

/// The velocity of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Velocity(pub Vec2);

/// The mass of a node.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Mass(pub f32);

impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}

/// The degree of a node, i.e., how many edges it has.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Degree(pub f32);

/// A fixed node skips all force computation.
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Fixed;

/// A dragged node has force applied by the user.
///
/// It skips all force computation.
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Dragged;

/// A shown node is visible on screen.
///
/// A hidden node skips all force compute to save resources.
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Shown;

#[derive(Component)]
#[storage(VecStorage)]
pub struct NodeType(pub ElementType);

impl Default for NodeType {
    fn default() -> Self {
        Self(ElementType::NoDraw)
    }
}
