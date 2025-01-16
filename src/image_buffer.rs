// use std::sync::{Arc, Mutex};
//
// struct ImageBuffer {
//     buf0: Box<[u32]>,
//     buf1: Box<[u32]>,
//
//     buf0_borrowed: bool,
//     buf1_borrowed: bool,
//
//     draw_buf0: bool,
// }
//
// struct ImageBufferHandle {
//     is_handle0: bool,
//     buffer: Arc<Mutex<ImageBuffer>>,
// }
//
// impl ImageBuffer {
//     fn new(w: usize, h: usize) -> Self {
//         let this = Self {
//             buf0: vec![0; w * h].into_boxed_slice(),
//             buf1: vec![0; w * h].into_boxed_slice(),
//
//             buf0_borrowed: true,
//             buf1_borrowed: true,
//
//             draw_buf0: false,
//         }
//
//
//     }
// }
