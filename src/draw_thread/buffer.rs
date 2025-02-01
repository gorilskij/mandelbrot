use crate::rendering::CoordinatesBox;
use parking_lot::Mutex;
use std::sync::Arc;

pub struct SubBuffer {
    pub buffer: Box<[u32]>,
    pub coords: CoordinatesBox,
}

impl SubBuffer {
    fn new(width: usize, height: usize, coords: CoordinatesBox) -> Self {
        Self {
            buffer: vec![0; width * height].into_boxed_slice(),
            coords,
        }
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub base: Arc<Mutex<SubBuffer>>,
    pub zoomed: Arc<Mutex<SubBuffer>>,
}

impl Buffer {
    pub fn new(width: usize, height: usize, coords: CoordinatesBox) -> Self {
        Self {
            base: Arc::new(Mutex::new(SubBuffer::new(width, height, coords))),
            zoomed: Arc::new(Mutex::new(SubBuffer::new(width, height, coords))),
        }
    }
}
