use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Vec2;
use grapher::prelude::Simulator;

/// Force-directed graph simulation speed
fn force_directed_simulation(c: &mut Criterion) {
    let nodes: Vec<u32> = vec![3_000, 30_000, 300_000, 3_000_000, 16_777_216]; // Max size of hibitset is 16777216
    let edges: Vec<[u32; 2]> = vec![[0_u32, 0_u32]];

    let mut group = c.benchmark_group("Force-directed");

    let mut positions = Vec::with_capacity(nodes[4] as usize);
    let mut sizes: Vec<f32> = Vec::with_capacity(nodes[4] as usize);
    for i in nodes {
        positions.clear();
        sizes.clear();
        for j in 0..i {
            positions.push(Vec2::new(
                f32::fract(f32::sin(j as f32 * 12_345.679)),
                f32::fract(f32::sin(j as f32 * 98_765.43)),
            ));
            sizes.push(1.0);
        }
        let mut simulator = Simulator::builder().build(&positions, &edges, &sizes);

        if i >= 3_000_000 {
            group.sample_size(10);
        }

        group.throughput(criterion::Throughput::Elements(u64::from(i)));
        group.bench_function(BenchmarkId::new("Graph", i), |b| {
            b.iter(|| {
                simulator.tick();
            });
        });
    }
    group.finish();
}

criterion_group!(simulation, force_directed_simulation);
criterion_main!(simulation);
