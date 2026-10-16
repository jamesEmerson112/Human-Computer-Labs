//! Plays one MELD test dialogue clip by clip and, as each clip ends, asks `serve.py` for Whisper's
//! words, the trained head's emotion and LFM2.5's reply, falling back to a random emotion and a
//! placeholder reply when a server is not answering.
//!
//! Contents:
//! - `main`: plays the chosen dialogue and prints one result per line
//! - `Options`: the parsed command line
//! - `Line`: one line of the dialogue
//! - `Models`: the connection to `serve.py` and its failure state
//! - `Models::new`: builds the HTTP agent for a base URL
//! - `Models::get`: sends one GET request and reads its JSON
//! - `Models::warn`: prints the run's single warning on the first failure
//! - `Models::transcribe`: asks for Whisper's words for a clip
//! - `Models::emotion`: asks the head for a line's emotion and confidence
//! - `Models::reply`: asks LFM2.5 for a reply given the dialogue so far
//! - `parse_options`: reads the dialogue, `--server` and `--no-play` from the command line
//! - `load_dialogue`: reads one dialogue's lines from a label file
//! - `cut_text`: shortens a line to the servers' length limit
//! - `play_clip`: plays one clip with ffplay and waits for it to end
//! - `random_emotion_index`: picks one of MELD's seven emotions at random
//! - `random_reply`: picks one of an emotion's hardcoded replies at random

#[allow(dead_code)]
#[path = "../constants.rs"]
mod constants;

use constants::{DEMO_DIALOGUE, EMOTIONS, REPLIES, TEST_CLIPS_DIR, TEST_CSV};
use serde_json::{Value, json};
use std::io;
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

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
 * Description: Where `serve.py` listens when `--server` is not given.
 */
const DEFAULT_SERVER: &str = "http://127.0.0.1:8765";

/**
 * Description: The longest line, in characters, that `serve_emotion.py` and `serve_reply.py`
 * accept, matching their `MAX_TEXT`.
 */
const MAX_TEXT: usize = 600;

/**
 * Description: The most earlier lines `serve_reply.py` accepts as history, matching its
 * `MAX_HISTORY`.
 */
const MAX_HISTORY: usize = 40;

/**
 * Description: How long to wait for a connection to `serve.py` before treating it as off.
 */
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/**
 * Description: How long one request may take in all, which covers a reply that waits for its
 * turn on the GPU.
 */
const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

/**
 * Description: The parsed command line.
 * Fields:
 * - `dialogue`: the test Dialogue_ID to play.
 * - `server`: the base URL of `serve.py`, without a trailing slash.
 * - `play`: whether to play each clip with ffplay before asking the models.
 */
struct Options {
    dialogue: u32,
    server:   String,
    play:     bool,
}

/**
 * Description: One line of a dialogue.
 * Fields:
 * - `utterance_id`: the line's position in its dialogue, counting from 0.
 * - `speaker`: who says the line.
 * - `text`: MELD's transcript of the line.
 */
struct Line {
    utterance_id: u32,
    speaker:      String,
    text:         String,
}

/**
 * Description: The connection to `serve.py`, which forwards emotion and reply requests to its
 * helper servers.
 * Fields:
 * - `agent`: the HTTP agent, with its timeouts set.
 * - `base`: the base URL of `serve.py`.
 * - `reachable`: false once a request could not reach the server, so later calls are skipped.
 * - `warned`: true once the run's single warning has been printed.
 */
struct Models {
    agent:     ureq::Agent,
    base:      String,
    reachable: bool,
    warned:    bool,
}

impl Models {
    /**
     * Description: Builds the HTTP agent for `serve.py`.
     * Inputs:
     * - `base`: the base URL of `serve.py`, such as `http://127.0.0.1:8765`.
     * Outputs: a `Models` that has not yet sent a request.
     * Constraints: error statuses are read as ordinary responses, so `get` can report the
     * server's own error message.
     */
    fn new(base: &str) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .build();
        Models {
            agent:     config.into(),
            base:      base.trim_end_matches('/').to_string(),
            reachable: true,
            warned:    false,
        }
    }

    /**
     * Description: Sends one GET request to `serve.py` and reads its JSON body. ureq
     * percent-encodes every query key and value.
     * Inputs:
     * - `path`: the endpoint, such as `/api/emotion`.
     * - `params`: the query parameters as key and value pairs.
     * Outputs: the JSON body and the round trip in milliseconds, or `None` when the server is
     * unreachable, answers with an error status, or sends a body that is not JSON.
     * Constraints: prints one warning per run, on the first failure. After a failure to connect,
     * every later call returns `None` without sending anything. Any other failure, such as a
     * timeout on a connected server, affects only this call.
     */
    fn get(&mut self, path: &str, params: &[(&str, &str)]) -> Option<(Value, u128)> {
        if !self.reachable {
            return None;
        }
        let started = Instant::now();
        let mut request = self.agent.get(format!("{}{path}", self.base));
        for (key, value) in params {
            request = request.query(*key, *value);
        }

        let mut response = match request.call() {
            Ok(response) => response,
            Err(error) => {
                // Only a failed connection means the server is off. A request that times out on
                // a connected server, such as a reply waiting for the GPU, fails on its own.
                let unreachable = matches!(
                    error,
                    ureq::Error::Io(_)
                        | ureq::Error::ConnectionFailed
                        | ureq::Error::HostNotFound
                        | ureq::Error::BadUri(_)
                        | ureq::Error::Http(_)
                        | ureq::Error::Timeout(ureq::Timeout::Connect | ureq::Timeout::Resolve)
                );
                if unreachable {
                    self.reachable = false;
                    self.warn(&format!("could not reach {} ({error})", self.base));
                } else {
                    self.warn(&format!("{path} failed ({error})"));
                }
                return None;
            }
        };
        let status  = response.status();
        let body    = response.body_mut().read_to_string().unwrap_or_default();
        let elapsed = started.elapsed().as_millis();

        match serde_json::from_str::<Value>(&body) {
            Ok(value) if status.is_success() => Some((value, elapsed)),
            Ok(value) => {
                let message = value["error"].as_str().unwrap_or("no message").to_string();
                self.warn(&format!("{path} answered {status}: {message}"));
                None
            }
            Err(error) => {
                self.warn(&format!("{path} sent a body that is not JSON ({error})"));
                None
            }
        }
    }

    /**
     * Description: Prints the run's single warning, the first time anything fails.
     * Inputs:
     * - `problem`: what went wrong.
     * Outputs: none
     */
    fn warn(&mut self, problem: &str) {
        if self.warned {
            return;
        }
        self.warned = true;
        eprintln!(
            "warning: {problem}. Missing results fall back to a random emotion, marked (random), \
             and a canned reply, marked (placeholder)."
        );
    }

    /**
     * Description: Asks `serve.py` for Whisper's words for one test clip.
     * Inputs:
     * - `clip`: the clip name, such as `dia149_utt0`.
     * Outputs: the words and the round trip in milliseconds, or `None` on a failure.
     */
    fn transcribe(&mut self, clip: &str) -> Option<(String, u128)> {
        let params = [("model", "whisper"), ("split", "test"), ("clip", clip)];
        let (body, elapsed) = self.get("/api/transcribe", &params)?;
        Some((body["text"].as_str()?.to_string(), elapsed))
    }

    /**
     * Description: Asks the trained head, through `serve.py`, for one test line's emotion.
     * Inputs:
     * - `clip`: the clip name, whose audio the head hears.
     * - `text`: the words the head reads.
     * Outputs: the emotion's index into `EMOTIONS`, its confidence and the round trip in
     * milliseconds, or `None` on a failure or an emotion name outside `EMOTIONS`.
     */
    fn emotion(&mut self, clip: &str, text: &str) -> Option<(usize, f64, u128)> {
        let params = [("split", "test"), ("clip", clip), ("text", text)];
        let (body, elapsed) = self.get("/api/emotion", &params)?;
        let name       = body["emotion"].as_str()?;
        let index      = EMOTIONS.iter().position(|emotion| *emotion == name)?;
        let confidence = body["confidence"].as_f64()?;
        Some((index, confidence, elapsed))
    }

    /**
     * Description: Asks LFM2.5, through `serve.py`, for a reply to one line given the dialogue
     * so far.
     * Inputs:
     * - `text`: the line to reply to.
     * - `emotion`: the line's emotion name.
     * - `speaker`: who says the line.
     * - `history`: the dialogue's earlier lines, oldest first, each a `{speaker, text}` object.
     * Outputs: the reply and the round trip in milliseconds, or `None` on a failure.
     * Constraints: sends only the last `MAX_HISTORY` earlier lines.
     */
    fn reply(
        &mut self,
        text:    &str,
        emotion: &str,
        speaker: &str,
        history: &[Value],
    ) -> Option<(String, u128)> {
        let recent  = &history[history.len().saturating_sub(MAX_HISTORY)..];
        let history = Value::from(recent.to_vec()).to_string();
        let params  = [
            ("text", text),
            ("emotion", emotion),
            ("speaker", speaker),
            ("history", history.as_str()),
        ];
        let (body, elapsed) = self.get("/api/reply", &params)?;
        Some((body["reply"].as_str()?.to_string(), elapsed))
    }
}

/**
 * Description: Plays every clip of the chosen test dialogue in order. When a clip ends, prints
 * its speaker and MELD's transcript, then Whisper's words, the head's emotion and LFM2.5's reply
 * with each step's round trip.
 * Inputs: none
 * Outputs: success once the dialogue has played, or failure when the command line is wrong,
 * the dialogue has no lines, or ffplay fails.
 * Constraints: panics when the label file cannot be read, so run it from
 * `ML_route/hello_world`. ffplay must be on the PATH unless `--no-play` is given. Any part a
 * server cannot give falls back to a random emotion or a placeholder reply, labelled as such.
 */
fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(options) => options,
        Err(problem) => {
            eprintln!("{problem}");
            eprintln!("usage: cargo run --bin demo -- [Dialogue_ID] [--server <url>] [--no-play]");
            return ExitCode::FAILURE;
        }
    };
    let dialogue = options.dialogue;
    let lines    = load_dialogue(TEST_CSV, dialogue);
    if lines.is_empty() {
        eprintln!("dialogue {dialogue} has no lines in {TEST_CSV}");
        return ExitCode::FAILURE;
    }
    println!("Playing test dialogue {dialogue}, {} lines, models at {}", lines.len(), options.server);

    let mut models  = Models::new(&options.server);
    let mut history = Vec::new();
    for line in &lines {
        let clip      = format!("dia{dialogue}_utt{}", line.utterance_id);
        let clip_path = Path::new(TEST_CLIPS_DIR).join(format!("{clip}.mp4"));
        if !clip_path.exists() {
            println!("{clip}  missing clip, skipped");
            continue;
        }
        if options.play {
            if let Err(error) = play_clip(&clip_path) {
                eprintln!("could not play {clip}: {error}");
                return ExitCode::FAILURE;
            }
        }
        println!("{clip}  {}: \"{}\"", line.speaker, line.text);

        // The head reads Whisper's words, as in training, and MELD's text only when Whisper is off.
        let heard = models.transcribe(&clip);
        let words = match &heard {
            Some((text, elapsed)) => {
                println!("    heard: \"{text}\" ({elapsed} ms)");
                cut_text(text)
            }
            None => {
                println!("    heard: (no transcript, using MELD's text)");
                cut_text(&line.text)
            }
        };

        let live_emotion = models.emotion(&clip, &words);
        let (emotion_index, emotion_label) = match live_emotion {
            Some((index, confidence, _)) => (index, format!("{} {confidence:.2}", EMOTIONS[index])),
            None => {
                let index = random_emotion_index();
                (index, format!("{} (random)", EMOTIONS[index]))
            }
        };

        // serve_reply.py refuses an empty line, so a silent clip is answered with MELD's text.
        let reply_text = if words.trim().is_empty() { cut_text(&line.text) } else { words };
        let live_reply = models.reply(&reply_text, EMOTIONS[emotion_index], &line.speaker, &history);
        let (reply, reply_label) = match &live_reply {
            Some((reply, _)) => (reply.as_str(), ""),
            None => (random_reply(emotion_index), " (placeholder)"),
        };

        let mut times = Vec::new();
        if let Some((_, _, elapsed)) = live_emotion {
            times.push(format!("emotion {elapsed} ms"));
        }
        if let Some((_, elapsed)) = live_reply {
            times.push(format!("reply {elapsed} ms"));
        }
        let times = if times.is_empty() { String::new() } else { format!(" ({})", times.join(", ")) };
        println!("    -> {emotion_label}: \"{reply}\"{reply_label}{times}");

        history.push(json!({ "speaker": line.speaker, "text": line.text }));
    }
    ExitCode::SUCCESS
}

/**
 * Description: Reads the command line: an optional Dialogue_ID, `--server <url>` and
 * `--no-play`, in any order.
 * Inputs: none
 * Outputs: the options, with `DEMO_DIALOGUE`, `DEFAULT_SERVER` and playback on by default, or
 * a message saying what is wrong.
 * Constraints: the `--server` URL must start with `http://`, since the HTTP client is built
 * without TLS.
 */
fn parse_options() -> Result<Options, String> {
    let mut options = Options {
        dialogue: DEMO_DIALOGUE,
        server:   DEFAULT_SERVER.to_string(),
        play:     true,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--no-play" => options.play = false,
            "--server" => {
                let url = arguments.next().ok_or("--server needs a URL")?;
                if !url.starts_with("http://") {
                    return Err(format!("--server needs an http:// URL, not {url}"));
                }
                options.server = url;
            }
            _ => {
                options.dialogue = argument
                    .parse()
                    .map_err(|_| format!("{argument} is not a Dialogue_ID or a known flag"))?;
            }
        }
    }
    Ok(options)
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
 * Description: Shortens a line to at most `MAX_TEXT` characters, cutting at the last space when
 * there is one, as the MELD player's `cutWords` does.
 * Inputs:
 * - `text`: the line.
 * Outputs: the line itself when it fits, or its shortened start.
 */
fn cut_text(text: &str) -> String {
    if text.chars().count() <= MAX_TEXT {
        return text.to_string();
    }
    let head: String = text.chars().take(MAX_TEXT).collect();
    match head.rfind(' ') {
        Some(space) if space > 0 => head[..space].to_string(),
        _ => head,
    }
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
 * for the trained head when the emotion server is not answering.
 * Inputs: none
 * Outputs: the emotion's index into `EMOTIONS`.
 */
fn random_emotion_index() -> usize {
    rand::random_range(0..EMOTIONS.len())
}

/**
 * Description: Picks one of an emotion's hardcoded replies at random. It
 * stands in for the reply model when the reply server is not answering.
 * Inputs:
 * - `emotion_index`: the emotion's index into `EMOTIONS`.
 * Outputs: one reply from that emotion's row of `REPLIES`.
 * Constraints: panics when `emotion_index` is 7 or more.
 */
fn random_reply(emotion_index: usize) -> &'static str {
    let replies = &REPLIES[emotion_index];
    replies[rand::random_range(0..replies.len())]
}
