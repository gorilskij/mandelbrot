pub mod maybe_pixel;
mod renderer;

use crate::{
    drawing::renderer::multithreaded,
    rendering::{CoordinatesBox, Pixels},
    support::Point,
    tiles::{compose::compose, store::MEMORY_BUDGET_BYTES, store::TileStore},
};
use std::{sync::Arc, thread};

/// Owns the tile store, the render thread handle and the display buffer.
/// All preview/caching logic that used to live here (cached_buf,
/// partial_bufs, resampling) is replaced by the tile store + compositor:
/// the compositor can assemble any viewport from whatever tiles exist, so
/// zooming and panning just change the coordinates that get composed.
pub struct Drawer {
    width: usize,
    height: usize,
    current_coords: CoordinatesBox,
    display_buf: Box<[u32]>,
    store: Arc<TileStore>,
    renderer: multithreaded::Handle,
    /// store progress at the time of the last compose
    last_progress: u64,
    /// view changed since the last compose
    dirty: bool,
}

impl Drawer {
    pub fn new(width: usize, height: usize, coords: CoordinatesBox, iterations: usize) -> Self {
        let store = Arc::new(TileStore::new(MEMORY_BUDGET_BYTES));
        let renderer = multithreaded::spawn(store.clone());

        let mut this = Self {
            width,
            height,
            current_coords: coords.clone(),
            display_buf: vec![0; width * height].into_boxed_slice(),
            store,
            renderer,
            last_progress: 0,
            dirty: true,
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
        self.dirty = true;
    }

    /// Resize the display buffer to match the window and re-render the current
    /// view at the new size.
    pub fn resize(&mut self, width: usize, height: usize, iterations: usize) {
        self.width = width;
        self.height = height;
        self.display_buf = vec![0; width * height].into_boxed_slice();
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

    /// Recompose the display buffer if the view moved or rendering made
    /// progress since the last call.
    pub fn update_display_buf(&mut self) {
        let progress = self.store.progress();
        if !self.dirty && progress == self.last_progress {
            return;
        }
        self.last_progress = progress;
        self.dirty = false;

        compose(
            &self.store,
            &self.current_coords,
            self.width,
            self.height,
            &mut self.display_buf,
        );
    }

    pub fn display_buf(&self) -> &[u32] {
        &self.display_buf
    }

    pub fn stop(self) -> thread::Result<()> {
        self.renderer.terminate_and_join()
    }
}
