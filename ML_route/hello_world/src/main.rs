//! Prints a greeting and one dialogue from the MELD training labels.
//!
//! Contents:
//! - `TRAIN_CSV`: path to the MELD training label file
//! - `DEFAULT_DIALOGUE`: the dialogue printed when none is given
//! - `main`: prints "Hello, World!", then the lines of the default dialogue
//! - `print_out_train_data`: prints the lines of one training dialogue as a table

/// Description: Path to MELD's training label file, relative to this crate's folder.
const TRAIN_CSV: &str = "../data/MELD.Raw/train_sent_emo.csv";

/// Description: The Dialogue_ID that `print_out_train_data` prints when it is given `None`.
const DEFAULT_DIALOGUE: u32 = 1036;

/// Description: Prints "Hello, World!", then the lines of the default dialogue.
/// Inputs: none
/// Outputs: none
fn main() {
    println!("Hello, World!");
    print_out_train_data(None);
}

/// Description: Prints every line of one dialogue in the MELD training file as a table of row
/// number, clip name, speaker, emotion, sentiment and utterance.
/// Inputs:
/// - `dialogue`: the Dialogue_ID to print, counting from 0, or `None` for `DEFAULT_DIALOGUE`.
/// Outputs: none. An ID with no lines prints only the table header.
/// Constraints: panics when the file cannot be read or a row cannot be parsed, so run it from
/// `ML_route/hello_world`.
fn print_out_train_data(dialogue: Option<u32>) {
    let dialogue = dialogue.unwrap_or(DEFAULT_DIALOGUE);
    let mut reader = csv::Reader::from_path(TRAIN_CSV).expect("could not read the MELD training file");
    println!("{:<5} {:<13} {:<16} {:<8} {:<9} {}", "No.", "Clip", "Speaker", "Emotion", "Sentiment", "Utterance");
    println!("{}", "-".repeat(80));
    for record in reader.records() {
        let row = record.expect("could not parse a row");
        if row[5] != dialogue.to_string() {
            continue;
        }
        let clip = format!("dia{}_utt{}", &row[5], &row[6]);
        // MELD stores the apostrophe as the control character U+0092.
        let utterance = row[1].replace('\u{92}', "'");
        println!("{:<5} {:<13} {:<16} {:<8} {:<9} {}", &row[0], clip, &row[2], &row[3], &row[4], utterance);
    }
}
