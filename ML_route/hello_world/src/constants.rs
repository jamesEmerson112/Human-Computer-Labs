//! Holds the file paths and dialogue lists that `main.rs` uses.
//!
//! APIs:
//! - `TRAIN_CSV`: path to the MELD training label file
//! - `DEV_CSV`: path to the MELD dev label file
//! - `OUTPUT_TXT`: path of the text file the dialogues are written to
//! - `DEFAULT_DIALOGUES`: the dialogues written when none are given
//! - `TEST_DIALOGUES`: five randomly picked dialogues to test with
//! - `CLIPS_DIR`: folder holding the MELD training clips
//! - `EMOTIONS`: MELD's seven emotion names in a fixed order
//! - `REPLIES`: three placeholder replies for each emotion

/**
 * Description: Path to the fixed MELD training label file in MELD.Completed,
 * relative to this crate's folder.
 */
pub const TRAIN_CSV: &str = "../data/MELD.Completed/train_sent_emo.csv";

/**
 * Description: Path to the fixed MELD dev label file in MELD.Completed,
 * relative to this crate's folder.
 */
pub const DEV_CSV: &str = "../data/MELD.Completed/dev_sent_emo.csv";

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
 * Description: Folder holding the MELD training clips, relative to this
 * crate's folder.
 */
pub const CLIPS_DIR: &str = "../data/MELD.Raw/train_splits";

/**
 * Description: MELD's seven emotion names. An emotion's class index is its
 * position in this list.
 */
pub const EMOTIONS: [&str; 7] = [
    "neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear",
];

/**
 * Description: Hardcoded replies that stand in for the reply model. Row `i`
 * holds the three replies for `EMOTIONS[i]`.
 */
// ML_route/player/player.html holds a copy of this list, so change both.
pub const REPLIES: [[&str; 3]; 7] = [
    [
        "Got it. Tell me more.",
        "Okay, I'm following you.",
        "I see. What happens next?",
    ],
    [
        "That's great to hear!",
        "Ha, I love that.",
        "You sound really happy about it.",
    ],
    [
        "Wait, really?",
        "Whoa, I did not see that coming.",
        "No way. Are you serious?",
    ],
    [
        "That sounds frustrating.",
        "I can tell you're upset. What happened?",
        "Okay, let's take a breath.",
    ],
    [
        "I'm sorry. That sounds hard.",
        "That must hurt. I'm here.",
        "Do you want to talk about it?",
    ],
    [
        "Ew, that sounds gross.",
        "Yikes. I wouldn't like that either.",
        "That's pretty unpleasant.",
    ],
    [
        "That sounds scary. Are you okay?",
        "Don't worry, we'll figure it out.",
        "Take it easy. You're safe here.",
    ],
];
