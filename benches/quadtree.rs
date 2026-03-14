#![expect(clippy::expect_used, reason = "Benching is allowed to panic")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Vec2;
use grapher::prelude::{BoundingBox2D, QuadTree};
use rand::prelude::*;
use std::hint::black_box;

/// Quadtree construction speed
fn quadtree_insert(c: &mut Criterion) {
    let w = 1000.0;
    let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
    let mut qt: QuadTree = QuadTree::new(bb);
    let mut rng = StdRng::seed_from_u64(42);
    let mut group = c.benchmark_group("QuadTree");
    group.bench_function("Insert", |b| {
        b.iter(|| {
            qt.insert(
                black_box(Vec2::new(
                    rng.random_range((-w / 2.0)..(w / 2.0)),
                    rng.random_range((-w / 2.0)..(w / 2.0)),
                )),
                rng.random_range(1.0..2000.0),
            )
            .expect("Insert should succeed");
        });
    });
    group.finish();
}

/// Barnes-Hut algorithm performance
fn quadtree_barnes_hut(c: &mut Criterion) {
    const THETA: f32 = 1.0;
    const REPEL_FORCE: f32 = -1e8;
    const NODES: [u32; 5] = [3_000, 30_000, 300_000, 3_000_000, 30_000_000];

    let w = 1000.0;
    let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
    let mut rng = StdRng::seed_from_u64(42);
    let mut group = c.benchmark_group("QuadTree");

    for i in NODES {
        let mut qt = QuadTree::new(bb.clone());
        let mut uid = (0..i).cycle();
        for j in 0..i {
            qt.insert_id(
                j,
                black_box(Vec2::new(
                    rng.random_range((-w / 2.0)..(w / 2.0)),
                    rng.random_range((-w / 2.0)..(w / 2.0)),
                )),
                rng.random_range(1.0..2000.0),
            )
            .expect("Insert should succeed");
        }

        group.throughput(criterion::Throughput::Elements(u64::from(i)));
        group.bench_function(BenchmarkId::new("Barnes-Hut", i), |b| {
            b.iter(|| {
                qt.approximate_forces_on_body(
                    black_box(uid.next().expect("should be Some")),
                    black_box(Vec2::new(
                        rng.random_range((-w / 2.0)..(w / 2.0)),
                        rng.random_range((-w / 2.0)..(w / 2.0)),
                    )),
                    1.0,
                    THETA,
                    REPEL_FORCE,
                )
            });
        });
    }
    group.finish();
}

/// Bounding box construction speed
fn bounding_box_sub_quadrant(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let bb = BoundingBox2D::new(Vec2::ZERO, 1000.0, 1000.0);
    let mut group = c.benchmark_group("QuadTree");
    group.bench_function(BenchmarkId::new("BB Sub_quadrant", 0), |b| {
        b.iter(|| {
            #[allow(unused_must_use)]
            bb.sub_quadrant(black_box(rng.random_range(0..=3)));
        });
    });
    group.finish();
}

criterion_group!(
    quadtree,
    quadtree_insert,
    quadtree_barnes_hut,
    bounding_box_sub_quadrant
);
criterion_main!(quadtree);
