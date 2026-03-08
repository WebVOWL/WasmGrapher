use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Vec2;
use grapher::prelude::{BoundingBox2D, QuadTree};
use rand::Rng;
use std::hint::black_box;

/// Quadtree construction speed
fn quadtree_insert(c: &mut Criterion) {
    let w = 1000.0;
    let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
    let mut qt: QuadTree = QuadTree::new(bb);
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("QuadTree");

    group.bench_function("Insert", |b| {
        b.iter(|| {
            #[expect(clippy::unwrap_used, reason = "Benching is allowed to panic")]
            qt.insert(
                black_box(Vec2::new(
                    rng.gen_range((-w / 2.0)..(w / 2.0)),
                    rng.gen_range((-w / 2.0)..(w / 2.0)),
                )),
                rng.gen_range(1.0..2000.0),
            )
            .unwrap();
        });
    });
    group.finish();
}

/// Barnes-Hut algorithm performance
fn quadtree_get_stack(c: &mut Criterion) {
    const THETA: f32 = 1.0;
    const REPEL_FORCE: f32 = 1e8;
    const NODES: [u32; 5] = [3_000, 30_000, 300_000, 3_000_000, 30_000_000];

    let w = 1000.0;
    let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("QuadTree");

    for i in NODES {
        let mut qt = QuadTree::new(bb.clone());
        for _ in 0..i {
            let v = Vec2::new(
                rng.gen_range((-w / 2.0)..(w / 2.0)),
                rng.gen_range((-w / 2.0)..(w / 2.0)),
            );
            #[expect(clippy::unwrap_used, reason = "Benching is allowed to panic")]
            qt.insert(black_box(v), rng.gen_range(1.0..2000.0)).unwrap();
        }

        group.throughput(criterion::Throughput::Elements(u64::from(i)));
        group.bench_function(BenchmarkId::new("Barnes-Hut", i), |b| {
            b.iter(|| {
                qt.barnes_hut(
                    black_box(Vec2::new(
                        rng.gen_range((-w / 2.0)..(w / 2.0)),
                        rng.gen_range((-w / 2.0)..(w / 2.0)),
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
    let bb = BoundingBox2D::new(Vec2::ZERO, 1000.0, 1000.0);
    let mut group = c.benchmark_group("QuadTree");
    group.bench_function(BenchmarkId::new("BB Sub_quadrant", 0), |b| {
        b.iter(|| {
            #[allow(unused_must_use)]
            bb.sub_quadrant(black_box(rand::thread_rng().gen_range(0..=3)));
        });
    });
    group.finish();
}

criterion_group!(
    simulation,
    quadtree_insert,
    quadtree_get_stack,
    bounding_box_sub_quadrant
);
criterion_main!(simulation);
