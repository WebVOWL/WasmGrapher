use std::collections::{HashMap, HashSet};

use crate::renderer::elements::element_type::ElementType;
use crate::simulator::components::edges::Connects;
use crate::simulator::components::nodes::{NodeType, Shown};
use crate::simulator::{
    components::nodes::{Dragged, Fixed, Position, Velocity},
    ressources::simulator_vars::{
        CursorPosition, Damping, DeltaTime, FreezeThreshold, PointIntersection,
    },
};
use glam::Vec2;
use log::info;
use rayon::prelude::*;
use specs::prelude::*;
use specs::{Entities, Join, ParJoin, Read, ReadStorage, WriteStorage, shred};

#[derive(SystemData)]
pub struct VisibilitySystemData<'a> {
    entities: Entities<'a>,
    node_types: ReadStorage<'a, NodeType>,
    connects: ReadStorage<'a, Connects>,
    updater: Read<'a, LazyUpdate>,
}

/// Show/hide entities
pub fn update_visibility(
    mut data: VisibilitySystemData,
    element_checks: HashMap<ElementType, bool>,
) {
    let should_hide = element_checks
        .into_iter()
        .filter(|&(_, checked)| !checked)
        .map(|(element, _)| element)
        .collect::<HashSet<_>>();

    /// Check all nodes having edges (incl. the edges)
    (&data.entities, &data.node_types, &data.connects)
        .par_join()
        .for_each(|(entity, node_type, connect)| {
            if should_hide.contains(&node_type.0) {
                data.updater.remove::<Shown>(entity);

                for edge in &connect.targets {
                    data.updater.remove::<Shown>(*edge);
                }
            } else {
                data.updater.insert(entity, Shown);
            }
        });

    /// Check solitary nodes (no edges).
    (&data.entities, &data.node_types, !&data.connects)
        .par_join()
        .for_each(|(entity, node_type, ())| {
            if should_hide.contains(&node_type.0) {
                data.updater.remove::<Shown>(entity);
            } else {
                data.updater.insert(entity, Shown);
            }
        });
}
