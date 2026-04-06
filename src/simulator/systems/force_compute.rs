use crate::{
    quadtree::QuadTree,
    simulator::{
        components::{
            edges::Connects,
            forces::NodeForces,
            nodes::{Degree, Dragged, Fixed, Mass, Position, Shown, Velocity},
        },
        ressources::simulator_vars::{
            DeltaTime, GravityForce, QuadTreeTheta, RepelForce, SpringNeutralLength,
            SpringStiffness,
        },
    },
};
use glam::Vec2;
use rayon::prelude::*;
use specs::{Entities, ParJoin, Read, ReadExpect, ReadStorage, System, WriteStorage};

pub struct ComputeNodeForce;

impl<'a> System<'a> for ComputeNodeForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        ReadStorage<'a, Shown>,
        WriteStorage<'a, NodeForces>,
        ReadExpect<'a, QuadTree>,
        Read<'a, QuadTreeTheta>,
        Read<'a, RepelForce>,
    );

    fn run(
        &mut self,
        (
            entities,
            positions,
            masses,
            fixed,
            dragged,
            shown,
            mut node_forces,
            quadtree,
            theta,
            repel_force,
        ): Self::SystemData,
    ) {
        (
            &*entities,
            &positions,
            &masses,
            &mut node_forces,
            &shown,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, pos, mass, node_forces, _, (), ())| {
                node_forces.0 = quadtree.approximate_forces_on_body(
                    entity.id(),
                    pos.0,
                    mass.0,
                    theta.0,
                    repel_force.0,
                );
            });
    }
}

/// Computes center gravity of the world.
/// All elements will gravitate towards this point.
pub struct ComputeGravityForce;

impl<'a> System<'a> for ComputeGravityForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        ReadStorage<'a, Shown>,
        WriteStorage<'a, NodeForces>,
        Read<'a, GravityForce>,
    );
    fn run(
        &mut self,
        (entities, positions, masses, fixed, dragged, shown, mut forces, gravity_force): Self::SystemData,
    ) {
        (
            &entities,
            &positions,
            &masses,
            &mut forces,
            &shown,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, pos, mass, force, _, (), ())| {
                force.0 += -pos.0 * mass.0 * gravity_force.0;
            });
    }
}

pub struct ApplyNodeForce;

impl<'a> System<'a> for ApplyNodeForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, NodeForces>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        ReadStorage<'a, Shown>,
        Read<'a, DeltaTime>,
    );

    fn run(
        &mut self,
        (entities, forces, mut velocities, masses, fixed, dragged,shown, delta_time): Self::SystemData,
    ) {
        (
            &entities,
            &forces,
            &mut velocities,
            &masses,
            &shown,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, force, velocity, mass, _, (), ())| {
                velocity.0 += force.0 / mass.0 * delta_time.0;
            });
    }
}

pub struct ComputeEdgeForces;

impl<'a> System<'a> for ComputeEdgeForces {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Connects>,
        WriteStorage<'a, NodeForces>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Shown>,
        Read<'a, SpringStiffness>,
        Read<'a, SpringNeutralLength>,
        ReadStorage<'a, Degree>,
    );

    fn run(
        &mut self,
        (
            entities,
            connections,
            mut forces,
            positions,
            shown,
            spring_stiffness,
            spring_neutral_length,
            degrees,
        ): Self::SystemData,
    ) {
        let positions_storage = &positions;

        let force_updates: Vec<(specs::Entity, Vec2)> =
            (&entities, &positions, &connections, &shown)
                .par_join()
                .fold(Vec::new, |mut acc, (entity, pos, connects, _)| {
                    let rb1 = entity;
                    for rb2 in &connects.targets {
                        // Look up the neighbor's position
                        if let Some(pos2_comp) = positions_storage.get(*rb2) {
                            let pos2 = pos2_comp.0;
                            let direction_vec = pos2 - pos.0;

                            let force_magnitude = spring_stiffness.0
                                * (direction_vec.length() - spring_neutral_length.0);

                            let du = degrees.get(rb1).map_or(1.0, |d| d.0);
                            let dv = degrees.get(*rb2).map_or(1.0, |d| d.0);
                            let w = if (du + dv > 200.0) {
                                1.0 / (du * dv).sqrt().max(1.0)
                            } else {
                                1.0
                            };

                            let spring_force =
                                (direction_vec.normalize_or(Vec2::ZERO) * -force_magnitude) * w;

                            acc.push((rb1, -spring_force));
                            acc.push((*rb2, spring_force));
                        }
                    }
                    acc
                })
                .reduce(Vec::new, |mut a, b| {
                    a.extend(b);
                    a
                });

        for (entity, force_vec) in force_updates {
            if let Some(force_comp) = forces.get_mut(entity) {
                force_comp.0 += force_vec;
            }
        }
    }
}
