//! Step 3.7: saves the trained head so the player can load it.
//!
//! APIs:
//! - `save_head`: writes the head's sizes and weights to a JSON file

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::step3_2_forward::ClassifierHead;

/**
 * Description: Writes the trained head's sizes and weights to disk as JSON, which Python can read
 * with `json.load` for the player.
 * Inputs:
 * - `head`: the weights.
 * - `path`: where to write them; missing folders are created.
 * Outputs: `Ok(())`, or the write error.
 */
pub(crate) fn save_head(head: &ClassifierHead, path: &str) -> std::io::Result<()> {
    if let Some(folder) = Path::new(path).parent() {
        std::fs::create_dir_all(folder)?;
    }
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer(writer, head).map_err(std::io::Error::other)
}
