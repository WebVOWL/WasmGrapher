use crate::simulator::{
    components::nodes::{Dragged, Fixed, Position, Shown, Velocity},
    ressources::simulator_vars::{
        CursorPosition, Damping, DeltaTime, FreezeThreshold, PointIntersection,
    },
};

use glam::Vec2;
use log::info;
use rayon::prelude::*;
use specs::prelude::*;
use specs::{Entities, Join, ParJoin, Read, ReadStorage, WriteStorage, shred};

pub struct UpdateNodePosition;

impl<'a> System<'a> for UpdateNodePosition {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Velocity>,
        WriteStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        ReadStorage<'a, Shown>,
        Read<'a, DeltaTime>,
        Read<'a, Damping>,
        Read<'a, FreezeThreshold>,
        Read<'a, LazyUpdate>,
    );

    fn run(
        &mut self,
        (
            entities,
            mut positions,
            mut velocities,
            mut fixed,
            dragged,
            shown,
            delta_time,
            damping,
            freeze_threshold,
            updater,
        ): Self::SystemData,
    ) {
        (&entities, &mut velocities, !&fixed, !&dragged, &shown)
            .par_join()
            .for_each(|(entity, velocity, (), (), _)| {
                // Automatically freeze/unfreeze based on velocity
                if velocity.0.abs().length() < freeze_threshold.0 {
                    updater.insert(entity, Fixed);
                    velocity.0 = Vec2::ZERO;
                }
            });

        (
            &entities,
            &mut positions,
            &mut velocities,
            !&fixed,
            !&dragged,
            &shown,
        )
            .par_join()
            .for_each(|(entity, pos, velocity, (), (), _)| {
                velocity.0 *= damping.0;
                pos.0 += velocity.0 * delta_time.0;

                // Safety bounds
                if pos.0.distance(Vec2::new(0.0, 0.0)) > 10_000_000.0 {
                    pos.0 = Vec2::new(0.0, 0.0);
                }
            });
    }
}

#[derive(SystemData)]
pub struct DragStartSystemData<'a> {
    entities: Entities<'a>,
    fixed: ReadStorage<'a, Fixed>,
    dragged: ReadStorage<'a, Dragged>,
    shown: ReadStorage<'a, Shown>,
    dragged_id: Read<'a, PointIntersection>,
    updater: Read<'a, LazyUpdate>,
}

/// A node is being dragged.
pub fn sys_drag_start(mut data: DragStartSystemData) {
    if data.dragged_id.0 >= 0 {
        #[expect(clippy::unwrap_used)]
        let dragged_entity = data.entities.entity(data.dragged_id.0.try_into().unwrap());

        // Prevent dragging invisible entities.
        if !data.shown.contains(dragged_entity) {
            return;
        }

        // Unfix everything
        (&data.entities, &data.fixed)
            .join()
            .for_each(|(entity, _)| {
                data.updater.remove::<Fixed>(entity);
            });

        // Set dragged state on specific node
        data.updater.insert(dragged_entity, Dragged);
    }
}

#[derive(SystemData)]
pub struct DragEndSystemData<'a> {
    entities: Entities<'a>,
    dragged: ReadStorage<'a, Dragged>,
    shown: ReadStorage<'a, Shown>,
    updater: Read<'a, LazyUpdate>,
}

/// A node is no longer being dragged.
pub fn sys_drag_end(mut data: DragEndSystemData) {
    for (entity, _, _) in (&data.entities, &data.dragged, &data.shown).join() {
        data.updater.remove::<Dragged>(entity);
    }
}

#[derive(specs::SystemData)]
pub struct DraggingSystemData<'a> {
    entities: Entities<'a>,
    dragged: ReadStorage<'a, Dragged>,
    shown: ReadStorage<'a, Shown>,
    positions: WriteStorage<'a, Position>,
    cursor_position: Read<'a, CursorPosition>,
}

/// The position of the dragged node has changed.
pub fn sys_dragging(mut data: DraggingSystemData) {
    for (_, pos, _, _) in (
        &data.entities,
        &mut data.positions,
        &data.dragged,
        &data.shown,
    )
        .join()
    {
        pos.0 = data.cursor_position.0;
    }
}
