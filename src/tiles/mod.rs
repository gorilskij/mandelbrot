//! Tree-based tiling: the complex plane is recursively split into squares
//! (a quadtree). Rendered tiles are cached and reused across zooms and pans;
//! the compositor assembles any viewport from the best available tiles,
//! falling back to coarser (or finer) levels while rendering catches up.

pub mod perturb;
pub mod render;
pub mod store;
