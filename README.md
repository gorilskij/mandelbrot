# Mandelbrot explorer

An interactive deep-zoom Mandelbrot viewer in Rust (winit + wgpu), native and on the web —
try it at [gorilskij.com/mandelbrot/](https://gorilskij.com/mandelbrot/) (a WebGPU browser).
Perturbation theory on the GPU: nucleus references, rebasing, bilinear approximation,
interior detection and an extended-exponent pipeline, for views far below `f64`.

```
cargo run --release
```

Documentation: [`docs/`](docs/README.md), starting with
[getting started](docs/tutorials/getting-started.md).
