//! Times every kernel this CPU supports on the non-ASCII inputs; ASCII input
//! returns from the ASCII scan before any kernel runs. Kept in its own binary
//! so these cases don't move the code of the other benchmarks.

mod inputs;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::__kernels;

fn bench_kernels(c: &mut Criterion) {
    let inputs: Vec<_> = inputs::all()
        .into_iter()
        .filter(|(_, input)| !input.is_ascii())
        .collect();
    let mut group = c.benchmark_group("kernel");
    for kernel in __kernels::available() {
        for (name, input) in &inputs {
            group.bench_function(BenchmarkId::new(*name, kernel.name), |b| {
                b.iter(|| (kernel.utf16_len)(black_box(input.as_bytes())));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
