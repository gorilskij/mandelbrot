pub mod append_only;
mod euclid;

pub use euclid::*;

// unused
// pub fn read_line_stdin() -> Result<Option<String>, ()> {
//     let mut line = String::new();
//     std::io::stdin().read_line(&mut line).map_err(|_| ())?;
//     let trimmed = line.trim();
//     if trimmed.is_empty() {
//         Ok(None)
//     } else {
//         Ok(Some(trimmed.to_string()))
//     }
// }
