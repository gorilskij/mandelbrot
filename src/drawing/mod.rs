pub mod maybe_pixel;
mod renderer;

use crate::{
    drawing::renderer::multithreaded,
    rendering::{CoordinatesBox, Pixels},
    support::Point,
    tiles::{perturb::Perturbator, store::MEMORY_BUDGET_BYTES, store::TileStore},
};
use std::sync::Arc;

/// Owns the tile store and the compute render thread. Display assembly lives
/// in the GPU compositor, which reads tiles from the store directly; the
/// Drawer just tracks the current view and (re)launches rendering for it, so
/// zooming and panning are a matter of changing coordinates.
pub struct Drawer {
    width: usize,
    height: usize,
    current_coords: CoordinatesBox,
    store: Arc<TileStore>,
    renderer: multithreaded::Handle,
}

impl Drawer {
    pub fn new(
        width: usize,
        height: usize,
        coords: CoordinatesBox,
        iterations: usize,
        backend: Arc<dyn Perturbator + Send + Sync>,
    ) -> Self {
        let store = Arc::new(TileStore::new(MEMORY_BUDGET_BYTES));
        let renderer = multithreaded::spawn(store.clone(), backend);

        let mut this = Self {
            width,
            height,
            current_coords: coords.clone(),
            store,
            renderer,
        };

        // initial render
        this.update(coords, iterations, None);
        this
    }

    /// Update the view and (re)launch rendering for it, prioritized around
    /// the cursor when given. Cancels any render in progress; finished tiles
    /// are kept.
    pub fn update(
        &mut self,
        new_coords: CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        self.renderer
            .update(&new_coords, iterations, cursor_rel, self.width, self.height);
        self.current_coords = new_coords;
    }

    /// Re-render the current view at a new window size.
    pub fn resize(&mut self, width: usize, height: usize, iterations: usize) {
        self.width = width;
        self.height = height;
        let coords = self.current_coords.clone();
        self.update(coords, iterations, None);
    }

    /// TEST: dump all tiles + reference lists and re-render from scratch
    /// (with fresh random reference orbits). Bound to spacebar.
    pub fn reset(
        &mut self,
        new_coords: CoordinatesBox,
        iterations: usize,
        cursor_rel: Option<&Point<usize, Pixels>>,
    ) {
        self.store.request_reset();
        self.update(new_coords, iterations, cursor_rel);
    }

    pub fn store(&self) -> &Arc<TileStore> {
        &self.store
    }

    /// Signal compute to stop and return immediately (the render thread is
    /// detached). Used on window close so exit is instant even mid-generation.
    pub fn stop(self) {
        self.renderer.terminate();
    }
}
