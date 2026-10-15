//! Plays one MELD training dialogue clip by clip and prints a random emotion
//! and a random reply for each line as soon as its clip ends.
//!
//! Contents:
//! - `main`: plays the chosen dialogue and prints one result per line
//! - `Line`: one line of the dialogue
//! - `parse_dialogue_id`: reads the Dialogue_ID from the command line
//! - `load_dialogue`: reads one dialogue's lines from a label file
//! - `play_clip`: plays one clip with ffplay and waits for it to end
//! - `random_emotion_index`: picks one of MELD's seven emotions at random
//! - `random_reply`: picks one of an emotion's hardcoded replies at random

#[allow(dead_code)]
#[path = "../constants.rs"]
mod constants;

use constants::{CLIPS_DIR, EMOTIONS, REPLIES, TEST_DIALOGUES, TRAIN_CSV};
use std::io;
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};

/**
 * Description: Column of the label file that holds the utterance text.
 */
const UTTERANCE_COLUMN: usize = 1;

/**
 * Description: Column of the label file that holds the speaker's name.
 */
const SPEAKER_COLUMN: usize = 2;

/**
 * Description: Column of the label file that holds the Dialogue_ID.
 */
const DIALOGUE_COLUMN: usize = 5;

/**
 * Description: Column of the label file that holds the Utterance_ID.
 */
const UTTERANCE_ID_COLUMN: usize = 6;

/**
 * Description: One line of a dialogue.
 * Fields:
 * - `utterance_id`: the line's position in its dialogue, counting from 0.
 * - `speaker`: who says the line.
 * - `text`: MELD's transcript of the line.
 */
struct Line {
    utterance_id: u32,
    speaker: String,
    text: String,
}

/**
 * Description: Plays every clip of the chosen dialogue in order. When a clip
 * ends, prints its speaker, its transcript, a random emotion and a random
 * reply for that emotion.
 * Inputs: none
 * Outputs: success once the dialogue has played, or failure when the
 * argument is not a number, the dialogue has no lines, or ffplay fails.
 * Constraints: panics when the label file cannot be read, so run it from
 * `ML_route/hello_world`. ffplay must be on the PATH.
 */
fn main() -> ExitCode {
    let Some(dialogue) = parse_dialogue_id() else {
        eprintln!("usage: cargo run --bin demo -- <Dialogue_ID>");
        return ExitCode::FAILURE;
    };
    let lines = load_dialogue(TRAIN_CSV, dialogue);
    if lines.is_empty() {
        eprintln!("dialogue {dialogue} has no lines in {TRAIN_CSV}");
        return ExitCode::FAILURE;
    }
    println!("Playing dialogue {dialogue}, {} lines", lines.len());

    for line in &lines {
        let clip      = format!("dia{dialogue}_utt{}", line.utterance_id);
        let clip_path = Path::new(CLIPS_DIR).join(format!("{clip}.mp4"));
        if !clip_path.exists() {
            println!("{clip}  missing clip, skipped");
            continue;
        }
        if let Err(error) = play_clip(&clip_path) {
            eprintln!("could not play {clip}: {error}");
            return ExitCode::FAILURE;
        }
        let emotion_index = random_emotion_index();
        let reply         = random_reply(emotion_index);
        println!("{clip}  {}: \"{}\"", line.speaker, line.text);
        println!("    -> {}: \"{reply}\"", EMOTIONS[emotion_index]);
    }
    ExitCode::SUCCESS
}

/**
 * Description: Reads the Dialogue_ID from the first command-line argument.
 * Inputs: none
 * Outputs: the ID, the first of `TEST_DIALOGUES` when no argument is given,
 * or `None` when the argument is not a whole number.
 */
fn parse_dialogue_id() -> Option<u32> {
    match std::env::args().nth(1) {
        None => Some(TEST_DIALOGUES[0]),
        Some(argument) => argument.parse().ok(),
    }
}

/**
 * Description: Reads every line of one dialogue from a MELD label file.
 * Inputs:
 * - `csv_path`: the label file.
 * - `dialogue`: the Dialogue_ID to keep.
 * Outputs: the dialogue's lines sorted by `utterance_id`, or an empty list
 * when the file has no such dialogue.
 * Constraints: panics when the file cannot be read or a row cannot be parsed.
 */
fn load_dialogue(csv_path: &str, dialogue: u32) -> Vec<Line> {
    let mut reader = csv::Reader::from_path(csv_path)
        .expect("could not read the MELD label file");
    let dialogue_text = dialogue.to_string();
    let mut lines     = Vec::new();

    for record in reader.records() {
        let row = record.expect("could not parse a row");
        if row[DIALOGUE_COLUMN] != dialogue_text {
            continue;
        }
        lines.push(Line {
            utterance_id: row[UTTERANCE_ID_COLUMN]
                .parse()
                .expect("Utterance_ID is not a number"),
            speaker: row[SPEAKER_COLUMN].to_string(),
            text: row[UTTERANCE_COLUMN].to_string(),
        });
    }
    lines.sort_by_key(|line| line.utterance_id);
    lines
}

/**
 * Description: Plays one clip in an ffplay window and waits until it ends.
 * Inputs:
 * - `clip_path`: the mp4 file to play.
 * Outputs: `Ok(())` once the clip has played, or an error when ffplay cannot
 * start or exits with a failure.
 */
fn play_clip(clip_path: &Path) -> io::Result<()> {
    let status = Command::new("ffplay")
        .args(["-autoexit", "-loglevel", "quiet"])
        .arg(clip_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!("ffplay exited with {status}")));
    }
    Ok(())
}

/**
 * Description: Picks one of MELD's seven emotions at random. It stands in
 * for the trained classifier.
 * Inputs: none
 * Outputs: the emotion's index into `EMOTIONS`.
 */
fn random_emotion_index() -> usize {
    rand::random_range(0..EMOTIONS.len())
}

/**
 * Description: Picks one of an emotion's hardcoded replies at random. It
 * stands in for the reply model.
 * Inputs:
 * - `emotion_index`: the emotion's index into `EMOTIONS`.
 * Outputs: one reply from that emotion's row of `REPLIES`.
 * Constraints: panics when `emotion_index` is 7 or more.
 */
fn random_reply(emotion_index: usize) -> &'static str {
    let replies = &REPLIES[emotion_index];
    replies[rand::random_range(0..replies.len())]
}
