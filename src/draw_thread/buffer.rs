use crate::rendering::{Point, Units, View};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct SubBuffer {
    pub buffer: Box<[u32]>,
    pub origin: Point<Units>,
    pub view: View,
}

impl SubBuffer {
    fn new(width: usize, height: usize, origin: Point<Units>, view: View) -> Self {
        Self {
            buffer: vec![0; width * height].into_boxed_slice(),
            origin,
            view,
        }
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub base: Arc<Mutex<SubBuffer>>,
    pub zoomed: Arc<Mutex<SubBuffer>>,
}

impl Buffer {
    pub fn new(width: usize, height: usize, origin: Point<Units>, view: View) -> Self {
        Self {
            base: Arc::new(Mutex::new(SubBuffer::new(width, height, origin, view))),
            zoomed: Arc::new(Mutex::new(SubBuffer::new(width, height, origin, view))),
        }
    }
}
