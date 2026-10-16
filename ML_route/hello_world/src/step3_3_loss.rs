//! Step 3.3: the cross-entropy loss that measures how wrong the head's scores are.
//!
//! APIs:
//! - `softmax`: turns scores into probabilities that sum to 1
//! - `cross_entropy`: the loss for one line's scores and its true emotion

/**
 * Description: Turns scores into probabilities that sum to 1. The largest score is subtracted
 * first so `exp` cannot overflow; this does not change the result.
 * Inputs:
 * - `logits`: the head's scores.
 * Outputs: one probability per emotion.
 */
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max  = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.iter().map(|&z| (z - max).exp()).collect::<Vec<f32>>();
    let sum  = exps.iter().sum::<f32>();
    exps.iter().map(|e| e / sum).collect()
}

/**
 * Description: Measures how wrong the scores are for the true emotion: minus the log of the
 * probability the head gives it. A perfect answer scores 0; a uniform guess over 7 scores ln 7.
 * Inputs:
 * - `logits`: the head's scores.
 * - `target`: the true emotion index.
 * Outputs: the cross-entropy loss.
 */
pub(crate) fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max     = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|&z| (z - max).exp()).sum::<f32>().ln() + max;
    log_sum - logits[target]
}
