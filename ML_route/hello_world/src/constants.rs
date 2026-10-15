//! Holds the file paths and dialogue lists that `main.rs` uses.
//!
//! APIs:
//! - `TRAIN_CSV`: path to the MELD training label file
//! - `DEV_CSV`: path to the MELD dev label file
//! - `OUTPUT_TXT`: path of the text file the dialogues are written to
//! - `DEFAULT_DIALOGUES`: the dialogues written when none are given
//! - `TEST_DIALOGUES`: five randomly picked dialogues to test with
//! - `MS_PER_SECOND`: milliseconds in one second

/**
 * Description: Path to the cleaned MELD training label file, relative to this crate's folder.
 */
pub const TRAIN_CSV: &str = "../data/MELD.Clean/train_sent_emo.csv";

/**
 * Description: Path to the cleaned MELD dev label file, relative to this crate's folder.
 */
pub const DEV_CSV: &str = "../data/MELD.Clean/dev_sent_emo.csv";

/**
 * Description: Path of the text file the dialogues are written to, relative to this crate's folder.
 */
pub const OUTPUT_TXT: &str = "dialogues.txt";

/**
 * Description: The Dialogue_IDs that `print_out_train_data` writes when it is given `None`.
 */
pub const DEFAULT_DIALOGUES: &[u32] = &[1036];

/**
 * Description: Five training Dialogue_IDs picked at random on 2026-10-14.
 */
pub const TEST_DIALOGUES: [u32; 5] = [557, 631, 700, 960, 963];

/**
 * Description: Milliseconds in one second, used to turn the training clock
 * and the time limit into the same unit.
 */
pub const MS_PER_SECOND: f32 = 1000.0;
