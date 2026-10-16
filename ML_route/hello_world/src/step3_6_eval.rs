//! Step 3.6: scores the head on a held-out split with loss and weighted F1.
//!
//! APIs:
//! - `eval_val`: the mean loss and weighted F1 on a split
//! - `weighted_f1`: weighted F1 from true and predicted labels

use crate::step3_1_load::Utterance;
use crate::step3_2_forward::{forward, ClassifierHead};
use crate::step3_3_loss::cross_entropy;

/**
 * Description: Scores the head on a held-out split, such as dev or test. Each line's prediction
 * is the emotion with the highest score.
 * Inputs:
 * - `head`: the weights.
 * - `dev`: the lines to score.
 * - `feature_dim`: feature vector length, checked against each line.
 * Outputs: the mean loss and the weighted F1 score, from 0 to 100.
 * Constraints: panics when a line's features are not `feature_dim` long.
 */
pub(crate) fn eval_val(head: &ClassifierHead, dev: &[Utterance], feature_dim: usize) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut truth      = Vec::with_capacity(dev.len());
    let mut predicted  = Vec::with_capacity(dev.len());

    for line in dev {
        assert_eq!(line.features.len(), feature_dim, "{}: wrong feature length", line.clip);
        let logits  = forward(head, &line.features);
        total_loss += cross_entropy(&logits, line.emotion);

        let best = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap_or(0);
        truth.push(line.emotion);
        predicted.push(best);
    }

    let mean_loss = total_loss / dev.len().max(1) as f32;
    (mean_loss, weighted_f1(&truth, &predicted, head.num_classes))
}

/**
 * Description: Weighted F1, the score MELD papers report: each emotion's F1 (the balance of its
 * precision and recall), averaged with weights equal to how many lines carry that emotion.
 * Inputs:
 * - `truth`: the true labels.
 * - `predicted`: the predicted labels, in the same order.
 * - `num_classes`: number of emotions.
 * Outputs: weighted F1 times 100.
 */
pub(crate) fn weighted_f1(truth: &[usize], predicted: &[usize], num_classes: usize) -> f32 {
    let mut true_pos  = vec![0usize; num_classes];
    let mut pred_cnt  = vec![0usize; num_classes];
    let mut truth_cnt = vec![0usize; num_classes];
    for (&t, &p) in truth.iter().zip(predicted) {
        truth_cnt[t] += 1;
        pred_cnt[p]  += 1;
        if t == p {
            true_pos[t] += 1;
        }
    }

    let mut weighted = 0.0;
    for c in 0..num_classes {
        let precision  = if pred_cnt[c] > 0 { true_pos[c] as f32 / pred_cnt[c] as f32 } else { 0.0 };
        let recall     = if truth_cnt[c] > 0 { true_pos[c] as f32 / truth_cnt[c] as f32 } else { 0.0 };
        let f1         = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        weighted      += f1 * truth_cnt[c] as f32;
    }
    100.0 * weighted / truth.len().max(1) as f32
}
