//! Step 3.1: loads a MELD split into labelled lines, each with its joined encoder features.
//!
//! APIs:
//! - `Utterance`: one MELD line with its label and features
//! - `load_split`: reads one label file and attaches each clip's features
//! - `emotion_to_index`: maps an emotion name to its class index

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

use crate::constants::EMOTIONS;

/**
 * Description: One MELD line with its label and features.
 * Fields:
 * - `clip`: clip name such as `dia47_utt1`.
 * - `speaker`: who says the line.
 * - `text`: the utterance.
 * - `emotion`: the label as an index into the emotion list.
 * - `features`: the clip's joined encoder features, `feature_dim` long.
 */
pub(crate) struct Utterance {
    pub(crate) clip:     String,
    pub(crate) speaker:  String,
    pub(crate) text:     String,
    pub(crate) emotion:  usize,
    pub(crate) features: Vec<f32>,
}

/**
 * Description: Reads one MELD label file and attaches each clip's joined features from
 * `prepare_features.py`'s `x_<split>.npy`, matched by clip name through `clips_<split>.txt`.
 * Inputs:
 * - `csv_path`: the label file.
 * - `features_dir`: the folder holding the joined feature files.
 * - `split`: "train", "dev" or "test", naming the feature files.
 * Outputs: every line that has features, in label-file order.
 * Constraints: panics when a file cannot be read, a row cannot be parsed, a row names an emotion
 * that is not in `EMOTIONS`, or a clip's label in the feature files disagrees with the CSV.
 */
pub(crate) fn load_split(csv_path: &str, features_dir: &str, split: &str) -> Vec<Utterance> {
    let x_path     = format!("{features_dir}/x_{split}.npy");
    let y_path     = format!("{features_dir}/y_{split}.npy");
    let clips_path = format!("{features_dir}/clips_{split}.txt");

    let x: Array2<f32> = read_npy(&x_path).unwrap_or_else(|e| panic!("could not read {x_path}: {e}"));
    let y: Array1<i64> = read_npy(&y_path).unwrap_or_else(|e| panic!("could not read {y_path}: {e}"));
    let clip_names     = std::fs::read_to_string(&clips_path)
        .unwrap_or_else(|e| panic!("could not read {clips_path}: {e}"));
    let row_of: HashMap<&str, usize> = clip_names.lines().enumerate().map(|(i, c)| (c, i)).collect();

    let mut reader = csv::Reader::from_path(csv_path)
        .expect("could not read the MELD label file");
    let mut lines   = Vec::new();
    let mut dropped = 0;

    for record in reader.records() {
        let row     = record.expect("could not parse a row");
        let clip    = format!("dia{}_utt{}", &row[5], &row[6]);
        let emotion = emotion_to_index(&row[3]);

        let Some(&i) = row_of.get(clip.as_str()) else {
            dropped += 1;
            continue;
        };
        assert_eq!(y[i] as usize, emotion, "{clip}: the feature files and {csv_path} disagree on its label");

        lines.push(Utterance {
            clip,
            speaker: row[2].to_string(),
            // MELD stores the apostrophe as the control character U+0092.
            text: row[1].replace('\u{92}', "'"),
            emotion,
            features: x.row(i).to_vec(),
        });
    }

    if dropped > 0 {
        println!("{split}: {dropped} lines have no features and are left out");
    }
    lines
}

/**
 * Description: Maps an emotion name to its class index, like `build_sentencepiece_luts` maps tokens.
 * Inputs:
 * - `emotion`: one of MELD's seven emotion names.
 * Outputs: the index from 0 to 6, its position in `EMOTIONS`.
 * Constraints: panics when the name is not one of the seven.
 */
pub(crate) fn emotion_to_index(emotion: &str) -> usize {
    EMOTIONS
        .iter()
        .position(|&name| name == emotion)
        .unwrap_or_else(|| panic!("unknown emotion: {emotion}"))
}
