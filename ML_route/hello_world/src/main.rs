//! Prints a greeting and writes a few dialogues from the MELD training labels to a text file.
//!
//! Contents:
//! - `main`: prints "Hello, World!", writes the five test dialogues, then runs the training batches
//! - `load_hyperparameters`: reads the training settings from `hyperparameters.toml`
//! - `default_hyperparameters`: the settings used for anything the file leaves out
//! - `print_out_train_data`: writes each chosen training dialogue as a table to `OUTPUT_TXT`
//! - `Hyperparameters`: the training settings
//! - `step3_1_load` to `step3_7_save`: one module per training lesson, in order

// The training placeholders are not wired up yet, so their warnings are silenced.
#![allow(dead_code, unused_variables, unused_imports)]

mod constants;
mod step3_1_load;
mod step3_2_forward;
mod step3_3_loss;
mod step3_4_update;
mod step3_5_loop;
mod step3_6_eval;
mod step3_7_save;

use constants::{
    DEFAULT_DIALOGUES, DEV_CSV, FEATURES_DIR, HEAD_JSON, HYPERPARAMETERS_TOML, OUTPUT_TXT, TEST_CSV,
    TEST_DIALOGUES, TRAIN_CSV,
};
use serde::Deserialize;
use step3_1_load::load_split;
use step3_2_forward::{forward, init_weights};
use step3_3_loss::cross_entropy;
use step3_4_update::{adamw_step, backward, lr_mul, zero_grad_all, AdamState};
use step3_5_loop::next_batch;
use step3_6_eval::eval_val;
use step3_7_save::save_head;
use std::fs::File;
use std::io::{BufWriter, Write};

// The chosen combo for the MELD demo (6B parameter limit; 3.11B with both backups loaded, 2.65B without):
// - Speech to text:  Whisper large-v3-turbo, int8     808.9M
// - Text encoder:    EmoBERTa-large (trained on MELD)  355.4M  first choice; likely saw MELD test lines
// - Text backup:     EmbeddingGemma-300m              302.9M
// - Voice encoder:   HuBERT-large                     315.4M
// - Voice backup:    emotion2vec+ large               162.1M
// - Reply writer:    LFM2.5-1.2B-Instruct             1.17B   (LFM2.5-2.6B tried as a larger option)
// - Emotion head:    trained here on MELD             under 1M

// Steps to train the fusion head for this combo (2,816 -> 256 -> 7; voice is 1,024 wide either way):
// 1. Get the features: copy the Whisper-text features of EmbeddingGemma (768) and EmoBERTa-large (1,024)
//    and the voice features (1,024) for train, dev and test from the pod volume.
// 2. Join them for Rust: line the three up by clip and write x_<split>.npy (2,816 per line) and
//    y_<split>.npy (labels 0 to 6). This side-by-side join is the "fusion".
// 3. Train the head here, one lesson at a time:
//    3.1 Load the data                      step3_1_load.rs     load_split
//    3.2 The forward pass                   step3_2_forward.rs  init_weights, forward
//    3.3 Cross-entropy loss                 step3_3_loss.rs     cross_entropy
//    3.4 Gradients and the update step      step3_4_update.rs   zero_grad_all, backward, adamw_step, lr_mul
//    3.5 The loop: shuffle, keep best dev   step3_5_loop.rs     next_batch, plus the loop in main
//    3.6 Weighted F1                        step3_6_eval.rs     eval_val
//    3.7 Save the head                      step3_7_save.rs     save_head
// 4. Check it: score the head on test and compare it with the Python heads from the A40 run.
// 5. Use it live: the player runs the three encoders per clip, the head gives the emotion, and
//    LFM2.5 writes the reply for it.

/**
 * Description: Prints "Hello, World!", writes the five test dialogues to `OUTPUT_TXT`, trains the
 * emotion head on the joined features, keeps the version with the best dev score, scores it on
 * test and saves it.
 * Inputs: none
 * Outputs: none
 * Constraints: panics when a file cannot be read or written; run it from `ML_route/hello_world`
 * after `prepare_features.py`, ideally with `cargo run --release`.
 */
fn main() {
    println!("Hello, World!");                       // say hello
    print_out_train_data(Some(&TEST_DIALOGUES))      // write five test dialogues to dialogues.txt
        .expect("could not write the output file");  // stop if the file can't be written

    // [1] and [2] happen before this program runs: copy the features from the pod, then
    // join them into x_<split>.npy and y_<split>.npy with prepare_features.py.
    let hp         = load_hyperparameters(HYPERPARAMETERS_TOML);            // training settings for this run
    let train_data = load_split(&hp.train_csv, &hp.features_dir, "train");  // [3.1] training lines with features
    let dev_data   = load_split(&hp.dev_csv, &hp.features_dir, "dev");      // [3.1] dev lines for validation
    println!("loaded {} train lines and {} dev lines", train_data.len(), dev_data.len());

    let mut head      = init_weights(&hp);                                    // [3.2] the head's weights, random at first
    let mut grads     = init_weights(&hp);                                    // [3.4] gradient buffer, same shape
    let mut state     = AdamState { m: Vec::new(), v: Vec::new(), step: 0 };  // [3.4] AdamW's running averages
    let mut best_head = head.clone();                                         // [3.5] the best head so far by dev score
    let mut best_f1   = f32::NEG_INFINITY;                                    // [3.5] its dev weighted F1
    let mut best_step = 0;                                                    // [3.5] the step it came from

    for step in 0..hp.iterations {                                  // [3.5] one training step per pass
        let (batch_features, batch_labels) =                        // this step's inputs and labels
            next_batch(&train_data, hp.batch_size, step, hp.seed);  // [3.5] take the next shuffled batch

        zero_grad_all(&mut grads);                                            // [3.4] clear last step's gradients
        let mut train_loss = 0.0;                                             // running loss for this batch
        for (features, &label) in batch_features.iter().zip(&batch_labels) {  // each line with its emotion
            let logits  = forward(&head, features);                           // [3.2] score the seven emotions
            train_loss += cross_entropy(&logits, label);                      // [3.3] add how wrong they were
            backward(&head, features, &logits, label, &mut grads);            // [3.4] add this line's gradients
        }
        train_loss /= batch_labels.len() as f32;                              // average over the batch

        let lr = hp.lr * lr_mul(step, &hp);                              // [3.4] learning rate after warmdown
        adamw_step(&mut head, &grads, &mut state, lr, hp.weight_decay);  // [3.4] update the weights

        if step % hp.train_log_every == 0 {                                              // every few steps
            println!(                                                                    // print the training loss
                "step {step}/{} train_loss {train_loss:.4} lr {lr:.6}",
                hp.iterations
            );
        }
        let last_step = step + 1 == hp.iterations;                                       // always score the final weights
        if step % hp.val_loss_every == 0 || last_step {                                  // every validation interval
            let (val_loss, val_f1) = eval_val(&head, &dev_data, hp.feature_dim);         // [3.6] score on dev
            println!("step {step} val_loss {val_loss:.4} val_weighted_f1 {val_f1:.2}");  // print the dev scores
            if val_f1 > best_f1 {                                                        // a new best on dev?
                best_f1   = val_f1;                                                      // remember its score
                best_head = head.clone();                                                // keep a copy of these weights
                best_step = step;                                                        // and when they were reached
            }
        }
    }

    println!("best dev weighted F1 {best_f1:.2} at step {best_step}");

    let test_data            = load_split(&hp.test_csv, &hp.features_dir, "test");  // [4] the test lines
    let (test_loss, test_f1) = eval_val(&best_head, &test_data, hp.feature_dim);    // [4] score the best head
    println!(                                                                       // [4] report it
        "test: {} lines, loss {test_loss:.4}, weighted F1 {test_f1:.2}",
        test_data.len()
    );

    save_head(&best_head, &hp.head_path)     // [3.7] write the best head to disk
        .expect("could not save the head");  // stop if it can't be written
    println!("saved the head to {}", hp.head_path);

    // [5] happens in the player: it runs the three encoders on each clip, feeds the joined
    // features to this head for the emotion, and LFM2.5 writes the reply for that emotion.
}

/**
 * Description: Reads the training settings from a TOML file, using the defaults for any setting the
 * file leaves out, or for all of them when the file does not exist.
 * Inputs:
 * - `path`: the TOML file, such as `HYPERPARAMETERS_TOML`.
 * Outputs: the settings for this run.
 * Constraints: panics when the file exists but cannot be read, is not valid TOML, or names a
 * setting that `Hyperparameters` does not have.
 */
fn load_hyperparameters(path: &str) -> Hyperparameters {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            println!("No {path}; using the default settings");
            return default_hyperparameters();
        }
        Err(err) => panic!("could not read {path}: {err}"),
    };

    let hp: Hyperparameters = toml::from_str(&text)
        .unwrap_or_else(|err| panic!("{path} is not valid: {err}"));
    println!("Loaded the settings from {path}");
    hp
}

/**
 * Description: Lets serde fill any setting the TOML file leaves out from `default_hyperparameters`.
 * Methods:
 * - `default`: returns `default_hyperparameters()`.
 */
impl Default for Hyperparameters {
    fn default() -> Self {
        default_hyperparameters()
    }
}

/**
 * Description: Builds the default training settings for one run of the MELD head.
 * Inputs: none
 * Outputs: the settings, reading the label paths from `constants.rs`.
 */
fn default_hyperparameters() -> Hyperparameters {
    Hyperparameters {
        train_csv: TRAIN_CSV.to_string(),
        dev_csv: DEV_CSV.to_string(),
        test_csv: TEST_CSV.to_string(),
        features_dir: FEATURES_DIR.to_string(),
        head_path: HEAD_JSON.to_string(),
        run_id: "meld-head".to_string(),
        seed: 1337,
        val_loss_every: 100,
        train_log_every: 20,
        iterations: 2000,
        warmdown_iters: 200,
        batch_size: 32,
        lr: 1e-3,
        weight_decay: 0.01,
        feature_dim: 3840,
        hidden_dim: 256,
        num_classes: 7,
        leaky_slope: 0.01,
    }
}

/**
 * Description: Writes every line of each chosen dialogue in the MELD training file to
 * `OUTPUT_TXT`, one table per dialogue, with row number, clip name, speaker, emotion, sentiment
 * and utterance. The file is replaced on every run.
 * Inputs:
 * - `dialogues`: the Dialogue_IDs to write, counting from 0, or `None` for `DEFAULT_DIALOGUES`.
 * Outputs: `Ok(())` once the file is written, or the error from creating or writing it. An ID
 * with no lines writes only its table header.
 * Constraints: panics when the CSV cannot be read or a row cannot be parsed, so run it from
 * `ML_route/hello_world`.
 */
fn print_out_train_data(dialogues: Option<&[u32]>) -> std::io::Result<()> {
    let dialogues = dialogues.unwrap_or(DEFAULT_DIALOGUES);
    let mut out   = BufWriter::new(File::create(OUTPUT_TXT)?);

    for &dialogue in dialogues {
        let mut reader = csv::Reader::from_path(TRAIN_CSV)
            .expect("could not read the MELD training file");

        writeln!(out)?;
        writeln!(out, "Dialogue {dialogue}")?;
        writeln!(
            out,
            "{:<5} {:<13} {:<16} {:<8} {:<9} {}",
            "No.",
            "Clip",
            "Speaker",
            "Emotion",
            "Sentiment",
            "Utterance",
        )?;
        writeln!(out, "{}", "-".repeat(80))?;

        for record in reader.records() {
            let row = record.expect("could not parse a row");
            if row[5] != dialogue.to_string() {
                continue;
            }

            let clip = format!("dia{}_utt{}", &row[5], &row[6]);
            // MELD stores the apostrophe as the control character U+0092.
            let utterance = row[1].replace('\u{92}', "'");
            writeln!(
                out,
                "{:<5} {:<13} {:<16} {:<8} {:<9} {}",
                &row[0],
                clip,
                &row[2],
                &row[3],
                &row[4],
                utterance,
            )?;
        }
    }

    println!("Wrote {} dialogues to {OUTPUT_TXT}", dialogues.len());
    Ok(())
}

/**
 * Description: The training settings, in the order `parameter_golf`'s `Hyperparameters` lists them.
 * Fields:
 * - `train_csv`, `dev_csv`, `test_csv`: label files to train on, validate on and test on.
 * - `features_dir`: folder of the joined encoder features from `prepare_features.py`.
 * - `head_path`: where the trained head is saved.
 * - `run_id`: name used for the log and the saved head.
 * - `seed`: random seed for weight init and shuffling.
 * - `val_loss_every`: run validation every this many steps.
 * - `train_log_every`: print the training loss every this many steps.
 * - `iterations`: total training steps.
 * - `warmdown_iters`: steps over which the learning rate falls to zero at the end.
 * - `batch_size`: lines per training step.
 * - `lr`: peak learning rate.
 * - `weight_decay`: AdamW weight decay.
 * - `feature_dim`: length of one encoder feature vector.
 * - `hidden_dim`: width of the head's hidden layer.
 * - `num_classes`: number of emotions, 7 for MELD.
 * - `leaky_slope`: Leaky ReLU's slope for negative inputs; 0 gives a plain ReLU.
 */
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Hyperparameters {
    train_csv:       String,
    dev_csv:         String,
    test_csv:        String,
    features_dir:    String,
    head_path:       String,
    run_id:          String,
    seed:            u64,
    val_loss_every:  usize,
    train_log_every: usize,
    iterations:      usize,
    warmdown_iters:  usize,
    batch_size:      usize,
    lr:              f32,
    weight_decay:    f32,
    feature_dim:     usize,
    hidden_dim:      usize,
    num_classes:     usize,
    leaky_slope:     f32,
}
