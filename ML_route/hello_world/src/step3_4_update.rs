//! Step 3.4: gradients and the AdamW update that moves the head's weights to lower the loss.
//!
//! APIs:
//! - `AdamState`: AdamW's running averages
//! - `zero_grad_all`: clears the gradients before a step
//! - `backward`: adds one line's gradients
//! - `scale_grads`: multiplies every gradient by one factor
//! - `clip_grad_norm`: shrinks the gradients when their combined size is over a limit
//! - `adamw_step`: applies one AdamW update
//! - `ema_update`: moves the weight average a little toward the current weights
//! - `lr_mul`: the step-based warmdown multiplier for the learning rate

use crate::Hyperparameters;
use crate::step3_2_forward::{activate, leaky, pre_activations, ClassifierHead};
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
 * For softmax with cross-entropy the score gradient is `p - onehot(target)`. When the softcap is
 * on it is scaled by the cap's slope, `1 - (z / c)^2`. It then flows back through `W2` and through
 * the activation: Leaky ReLU's slope (1 where the unit was positive, `leaky_slope` where not),
 * times `2 * leaky(pre)` when the activation is squared.
 * Inputs:
 * - `head`: the weights.
 * - `features`: the input that produced `logits`.
 * - `logits`: the head's scores from `forward`, after any softcap.
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
    let pre           = pre_activations(head, features);
    let h             = pre.iter().map(|&p| activate(head, p)).collect::<Vec<f32>>();
    let mut d_logits  = softmax(logits);
    d_logits[target] -= 1.0;

    let cap = head.logit_softcap;
    if cap > 0.0 {
        // The softcap c * tanh(z / c) has slope 1 - (capped / c)^2.
        for (d, &capped) in d_logits.iter_mut().zip(logits) {
            *d *= 1.0 - (capped / cap) * (capped / cap);
        }
    }

    let mut d_hidden = vec![0.0; head.hidden_dim];
    for (k, &d) in d_logits.iter().enumerate() {
        grads.b2[k] += d;
        for j in 0..head.hidden_dim {
            let w2_index        = k * head.hidden_dim + j;
            grads.w2[w2_index] += d * h[j];
            d_hidden[j]        += d * head.w2[w2_index];
        }
    }

    for j in 0..head.hidden_dim {
        // Leaky ReLU's slope is 1 where the unit was active and leaky_slope where it was not.
        let leaky_slope = if pre[j] > 0.0 { 1.0 } else { head.leaky_slope };
        let square_gain = if head.square_activation { 2.0 * leaky(pre[j], head.leaky_slope) } else { 1.0 };
        let d_pre       = d_hidden[j] * leaky_slope * square_gain;
        grads.b1[j]    += d_pre;
        let row_start   = j * head.feature_dim;
        let row         = &mut grads.w1[row_start..row_start + head.feature_dim];
        for (g, &x) in row.iter_mut().zip(features) {
            *g += d_pre * x;
        }
    }
}

/**
 * Description: Multiplies every gradient by one factor, such as `1 / batch size` to turn batch
 * sums into batch means.
 * Inputs:
 * - `grads`: the gradients, changed in place.
 * - `factor`: the multiplier.
 * Outputs: none
 */
pub(crate) fn scale_grads(grads: &mut ClassifierHead, factor: f32) {
    for g in [&mut grads.w1, &mut grads.b1, &mut grads.w2, &mut grads.b2] {
        g.iter_mut().for_each(|x| *x *= factor);
    }
}

/**
 * Description: Measures the gradients' combined size (the L2 norm over every weight) and, when it
 * is above `max_norm`, scales them all down so it equals `max_norm`, as `parameter_golf`'s
 * `grad_clip_norm` does.
 * Inputs:
 * - `grads`: the gradients, changed in place when clipped.
 * - `max_norm`: the limit; 0 or less measures without clipping.
 * Outputs: the norm before clipping.
 */
pub(crate) fn clip_grad_norm(grads: &mut ClassifierHead, max_norm: f32) -> f32 {
    let squares = [&grads.w1, &grads.b1, &grads.w2, &grads.b2]
        .iter()
        .map(|g| g.iter().map(|x| x * x).sum::<f32>())
        .sum::<f32>();
    let norm    = squares.sqrt();

    let too_big = max_norm > 0.0 && norm > max_norm;
    if too_big {
        scale_grads(grads, max_norm / norm);
    }
    norm
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
    let fix1    = 1.0 - BETA1.powi(state.step as i32);
    let fix2    = 1.0 - BETA2.powi(state.step as i32);

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
 * Description: Moves the weight average a little toward the current weights:
 * `ema = decay * ema + (1 - decay) * head`, as `parameter_golf`'s `ema_decay` does. With decay
 * 0.9965 the average mostly reflects the last few hundred steps, which smooths out step-to-step noise.
 * Inputs:
 * - `ema`: the averaged head, changed in place.
 * - `head`: the current weights.
 * - `decay`: how much of the old average to keep each step.
 * Outputs: none
 */
pub(crate) fn ema_update(ema: &mut ClassifierHead, head: &ClassifierHead, decay: f32) {
    let pairs = [
        (&mut ema.w1, &head.w1),
        (&mut ema.b1, &head.b1),
        (&mut ema.w2, &head.w2),
        (&mut ema.b2, &head.b2),
    ];
    for (average, current) in pairs {
        for (a, &c) in average.iter_mut().zip(current) {
            *a = decay * *a + (1.0 - decay) * c;
        }
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
     * Description: Builds a tiny head with fixed weights for the gradient checks.
     * Inputs:
     * - `square_activation`: whether the hidden activation is squared.
     * - `logit_softcap`: the softcap, or 0 for none.
     * Outputs: the head.
     */
    fn tiny_head(square_activation: bool, logit_softcap: f32) -> ClassifierHead {
        ClassifierHead {
            feature_dim: 4,
            hidden_dim: 3,
            num_classes: 3,
            leaky_slope: 0.01,
            square_activation,
            logit_softcap,
            feature_mean: Vec::new(),
            feature_std: Vec::new(),
            w1: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.3, -0.1, 0.25, 0.15, -0.35, 0.4],
            b1: vec![0.05, 0.1, -0.02],
            w2: vec![0.2, -0.3, 0.4, -0.1, 0.25, 0.3, 0.15, -0.2, 0.1],
            b2: vec![0.0, 0.1, -0.1],
        }
    }

    /**
     * Description: Picks one weight of the head to nudge.
     * Inputs:
     * - `head`: the head.
     * - `layer`: 0 for `w1`, 1 for `w2`.
     * - `i`: the weight's index in that layer.
     * Outputs: a mutable reference to the weight.
     */
    fn weight_mut(head: &mut ClassifierHead, layer: usize, i: usize) -> &mut f32 {
        if layer == 0 { &mut head.w1[i] } else { &mut head.w2[i] }
    }

    /**
     * Description: Checks `backward` against numerical gradients: nudging one weight up and down by
     * `eps` must change the loss by about its gradient times `2 * eps`, for every weight in w1 and w2.
     * Inputs:
     * - `head`: the head to check, changed and restored during the check.
     * Outputs: none; panics on a mismatch.
     */
    fn check_gradients(mut head: ClassifierHead) {
        let x      = [0.5, -1.0, 0.8, 0.3];
        let target = 1;

        let mut grads = head.clone();
        zero_grad_all(&mut grads);
        let logits = forward(&head, &x);
        backward(&head, &x, &logits, target, &mut grads);

        let eps = 1e-3;
        for layer in 0..2 {
            let count = if layer == 0 { head.w1.len() } else { head.w2.len() };
            for i in 0..count {
                let saved = *weight_mut(&mut head, layer, i);
                *weight_mut(&mut head, layer, i) = saved + eps;
                let up = cross_entropy(&forward(&head, &x), target);
                *weight_mut(&mut head, layer, i) = saved - eps;
                let down = cross_entropy(&forward(&head, &x), target);
                *weight_mut(&mut head, layer, i) = saved;

                let numeric  = (up - down) / (2.0 * eps);
                let analytic = if layer == 0 { grads.w1[i] } else { grads.w2[i] };
                assert!((numeric - analytic).abs() < 1e-3, "layer {layer} [{i}]: {numeric} vs {analytic}");
            }
        }
    }

    /**
     * Description: Plain Leaky ReLU with no softcap.
     * Inputs: none
     * Outputs: none
     */
    #[test]
    fn backward_matches_numerical_gradients() {
        check_gradients(tiny_head(false, 0.0));
    }

    /**
     * Description: LeakyReLU² with a softcap, so the squared-activation and softcap gradients are
     * checked too. A small cap of 0.5 makes the cap's effect large enough to test.
     * Inputs: none
     * Outputs: none
     */
    #[test]
    fn backward_matches_with_square_and_softcap() {
        check_gradients(tiny_head(true, 0.5));
    }
}
