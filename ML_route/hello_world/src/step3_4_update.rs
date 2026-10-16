//! Step 3.4: gradients and the AdamW update that moves the head's weights to lower the loss.
//!
//! APIs:
//! - `AdamState`: AdamW's running averages
//! - `zero_grad_all`: clears the gradients before a step
//! - `backward`: adds one line's gradients
//! - `adamw_step`: applies one AdamW update
//! - `lr_mul`: the step-based warmdown multiplier for the learning rate

use crate::Hyperparameters;
use crate::step3_2_forward::{hidden, ClassifierHead};
use crate::step3_3_loss::softmax;

/**
 * Description: AdamW's first moment decay rate.
 */
const BETA1: f32 = 0.9;

/**
 * Description: AdamW's second moment decay rate.
 */
const BETA2: f32 = 0.999;

/**
 * Description: A small number that keeps AdamW from dividing by zero.
 */
const EPS: f32 = 1e-8;

/**
 * Description: AdamW's running averages for every weight in the head, laid out as w1, b1, w2,
 * then b2. They start empty and are sized on the first update.
 * Fields:
 * - `m`: first-moment estimates, the running average of each gradient.
 * - `v`: second-moment estimates, the running average of each squared gradient.
 * - `step`: number of updates applied so far.
 */
pub(crate) struct AdamState {
    pub(crate) m:    Vec<f32>,
    pub(crate) v:    Vec<f32>,
    pub(crate) step: usize,
}

/**
 * Description: Sets every gradient back to zero before the next step.
 * Inputs:
 * - `grads`: the gradients to clear.
 * Outputs: none
 */
pub(crate) fn zero_grad_all(grads: &mut ClassifierHead) {
    for g in [&mut grads.w1, &mut grads.b1, &mut grads.w2, &mut grads.b2] {
        g.fill(0.0);
    }
}

/**
 * Description: Works out how each weight should change to lower the loss, and adds it to `grads`.
 * For softmax with cross-entropy the score gradient is `p - onehot(target)`; it flows back through
 * `W2`, then through the ReLU, which passes it only where the hidden unit was positive.
 * Inputs:
 * - `head`: the weights.
 * - `features`: the input that produced `logits`.
 * - `logits`: the head's scores.
 * - `target`: the true emotion index.
 * - `grads`: gradients, same shape as the head, added to in place.
 * Outputs: none
 */
pub(crate) fn backward(
    head: &ClassifierHead,
    features: &[f32],
    logits: &[f32],
    target: usize,
    grads: &mut ClassifierHead,
) {
    let h            = hidden(head, features);
    let mut d_logits = softmax(logits);
    d_logits[target] -= 1.0;

    let mut d_hidden = vec![0.0; head.hidden_dim];
    for (k, &d) in d_logits.iter().enumerate() {
        grads.b2[k] += d;
        for j in 0..head.hidden_dim {
            grads.w2[k * head.hidden_dim + j] += d * h[j];
            d_hidden[j]                       += d * head.w2[k * head.hidden_dim + j];
        }
    }

    for j in 0..head.hidden_dim {
        // ReLU passes the gradient only where the unit was active.
        if h[j] <= 0.0 {
            continue;
        }
        grads.b1[j] += d_hidden[j];
        let row      = &mut grads.w1[j * head.feature_dim..(j + 1) * head.feature_dim];
        for (g, &x) in row.iter_mut().zip(features) {
            *g += d_hidden[j] * x;
        }
    }
}

/**
 * Description: Applies one AdamW update to the head: each weight moves against its gradient,
 * scaled by the running averages, and the weight matrices also shrink by `weight_decay`.
 * Inputs:
 * - `head`: the weights to update.
 * - `grads`: this step's gradients.
 * - `state`: AdamW's running averages.
 * - `lr`: learning rate for this step.
 * - `weight_decay`: how strongly weights shrink each step.
 * Outputs: none
 * Constraints: Adam divides by each gradient's running size, so summed and averaged batch
 * gradients give the same step. Biases are not decayed.
 */
pub(crate) fn adamw_step(
    head: &mut ClassifierHead,
    grads: &ClassifierHead,
    state: &mut AdamState,
    lr: f32,
    weight_decay: f32,
) {
    let total = head.w1.len() + head.b1.len() + head.w2.len() + head.b2.len();
    if state.m.len() != total {
        state.m = vec![0.0; total];
        state.v = vec![0.0; total];
    }
    state.step += 1;
    let fix1 = 1.0 - BETA1.powi(state.step as i32);
    let fix2 = 1.0 - BETA2.powi(state.step as i32);

    let params = [
        (&mut head.w1, &grads.w1, weight_decay),
        (&mut head.b1, &grads.b1, 0.0),
        (&mut head.w2, &grads.w2, weight_decay),
        (&mut head.b2, &grads.b2, 0.0),
    ];
    let mut offset = 0;
    for (weights, g, decay) in params {
        for i in 0..weights.len() {
            let m = &mut state.m[offset + i];
            let v = &mut state.v[offset + i];
            *m    = BETA1 * *m + (1.0 - BETA1) * g[i];
            *v    = BETA2 * *v + (1.0 - BETA2) * g[i] * g[i];

            let step    = (*m / fix1) / ((*v / fix2).sqrt() + EPS);
            weights[i] -= lr * (step + decay * weights[i]);
        }
        offset += weights.len();
    }
}

/**
 * Description: Scales the learning rate down over the last `warmdown_iters` steps, counted in steps only.
 * Inputs:
 * - `step`: current step.
 * - `hp`: iterations and warmdown length.
 * Outputs: a multiplier from 0 to 1: 1 until the warmdown starts, then falling linearly to 0.
 */
pub(crate) fn lr_mul(step: usize, hp: &Hyperparameters) -> f32 {
    let warmdown_start = hp.iterations.saturating_sub(hp.warmdown_iters);
    if step < warmdown_start || hp.warmdown_iters == 0 {
        return 1.0;
    }
    (hp.iterations - step) as f32 / hp.warmdown_iters as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step3_2_forward::forward;
    use crate::step3_3_loss::cross_entropy;

    /**
     * Description: Checks `backward` against numerical gradients on a tiny head: nudging one weight
     * up and down by `eps` must change the loss by about its gradient times `2 * eps`.
     * Inputs: none
     * Outputs: none
     */
    #[test]
    fn backward_matches_numerical_gradients() {
        let mut head = ClassifierHead {
            feature_dim: 4,
            hidden_dim:  3,
            num_classes: 3,
            w1: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.3, -0.1, 0.25, 0.15, -0.35, 0.4],
            b1: vec![0.05, 0.1, -0.02],
            w2: vec![0.2, -0.3, 0.4, -0.1, 0.25, 0.3, 0.15, -0.2, 0.1],
            b2: vec![0.0, 0.1, -0.1],
        };
        let x      = [0.5, -1.0, 0.8, 0.3];
        let target = 1;

        let mut grads = head.clone();
        zero_grad_all(&mut grads);
        let logits = forward(&head, &x);
        backward(&head, &x, &logits, target, &mut grads);

        let eps = 1e-3;
        for i in 0..head.w1.len() {
            let saved   = head.w1[i];
            head.w1[i]  = saved + eps;
            let up      = cross_entropy(&forward(&head, &x), target);
            head.w1[i]  = saved - eps;
            let down    = cross_entropy(&forward(&head, &x), target);
            head.w1[i]  = saved;
            let numeric = (up - down) / (2.0 * eps);
            assert!((numeric - grads.w1[i]).abs() < 1e-3, "w1[{i}]: {numeric} vs {}", grads.w1[i]);
        }
        for i in 0..head.w2.len() {
            let saved   = head.w2[i];
            head.w2[i]  = saved + eps;
            let up      = cross_entropy(&forward(&head, &x), target);
            head.w2[i]  = saved - eps;
            let down    = cross_entropy(&forward(&head, &x), target);
            head.w2[i]  = saved;
            let numeric = (up - down) / (2.0 * eps);
            assert!((numeric - grads.w2[i]).abs() < 1e-3, "w2[{i}]: {numeric} vs {}", grads.w2[i]);
        }
    }
}
