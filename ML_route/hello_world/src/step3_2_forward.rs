//! Step 3.2: the head's weights and its forward pass from features to emotion scores.
//!
//! APIs:
//! - `ClassifierHead`: the head's weights and the settings that shape its forward pass
//! - `init_weights`: creates the head with random starting weights
//! - `pre_activations`: the first layer's output before its activation
//! - `leaky`: Leaky ReLU on one value
//! - `activate`: the hidden activation, Leaky ReLU or LeakyReLU²
//! - `hidden`: the first layer's output after its activation
//! - `forward`: runs the head on one feature vector

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

use crate::Hyperparameters;

/**
 * Description: The classification head's weights, two linear layers from features to emotions,
 * with the settings its forward pass needs. Each weight matrix is stored row by row in one flat
 * vector: row `j` of `w1` is `w1[j * feature_dim .. (j + 1) * feature_dim]`.
 * Fields:
 * - `feature_dim`, `hidden_dim`, `num_classes`: the layer sizes.
 * - `leaky_slope`: the hidden layer's slope for negative inputs; 0 makes it a plain ReLU.
 * - `square_activation`: squares the Leaky ReLU output (LeakyReLU², from `parameter_golf`).
 * - `logit_softcap`: caps the scores at about plus or minus this value with `c * tanh(z / c)`;
 *   0 turns it off.
 * - `feature_mean`, `feature_std`: the standardisation fitted on train, empty when it is off;
 *   saved so the player can scale live features the same way.
 * - `w1`, `b1`: first layer, `feature_dim` to `hidden_dim`.
 * - `w2`, `b2`: second layer, `hidden_dim` to `num_classes`.
 */
#[derive(Clone, Serialize)]
pub(crate) struct ClassifierHead {
    pub(crate) feature_dim:       usize,
    pub(crate) hidden_dim:        usize,
    pub(crate) num_classes:       usize,
    pub(crate) leaky_slope:       f32,
    pub(crate) square_activation: bool,
    pub(crate) logit_softcap:     f32,
    pub(crate) feature_mean:      Vec<f32>,
    pub(crate) feature_std:       Vec<f32>,
    pub(crate) w1:                Vec<f32>,
    pub(crate) b1:                Vec<f32>,
    pub(crate) w2:                Vec<f32>,
    pub(crate) b2:                Vec<f32>,
}

/**
 * Description: Creates the head with He-uniform random weights and zero biases, seeded by
 * `hp.seed` so every run starts from the same weights.
 * Inputs:
 * - `hp`: the sizes, activation settings and seed.
 * Outputs: the new head, with no standardisation recorded yet.
 */
pub(crate) fn init_weights(hp: &Hyperparameters) -> ClassifierHead {
    let mut rng = StdRng::seed_from_u64(hp.seed);
    // He-uniform keeps the signal's size steady through a (Leaky) ReLU layer.
    let gain    = 6.0 / (1.0 + hp.leaky_slope * hp.leaky_slope);
    let bound1  = (gain / hp.feature_dim as f32).sqrt();
    let bound2  = (gain / hp.hidden_dim as f32).sqrt();

    let w1 = (0..hp.hidden_dim * hp.feature_dim)
        .map(|_| rng.random_range(-bound1..bound1))
        .collect();
    let w2 = (0..hp.num_classes * hp.hidden_dim)
        .map(|_| rng.random_range(-bound2..bound2))
        .collect();

    ClassifierHead {
        feature_dim: hp.feature_dim,
        hidden_dim: hp.hidden_dim,
        num_classes: hp.num_classes,
        leaky_slope: hp.leaky_slope,
        square_activation: hp.square_activation,
        logit_softcap: hp.logit_softcap,
        feature_mean: Vec::new(),
        feature_std: Vec::new(),
        w1,
        b1: vec![0.0; hp.hidden_dim],
        w2,
        b2: vec![0.0; hp.num_classes],
    }
}

/**
 * Description: Runs the first layer without its activation: `W1 x + b1`.
 * Inputs:
 * - `head`: the weights.
 * - `features`: one feature vector, `feature_dim` long.
 * Outputs: the pre-activations, `hidden_dim` long.
 */
pub(crate) fn pre_activations(head: &ClassifierHead, features: &[f32]) -> Vec<f32> {
    (0..head.hidden_dim)
        .map(|j| {
            let row = &head.w1[j * head.feature_dim..(j + 1) * head.feature_dim];
            head.b1[j] + row.iter().zip(features).map(|(w, x)| w * x).sum::<f32>()
        })
        .collect()
}

/**
 * Description: Leaky ReLU on one value: the value where positive, `slope` times it where not.
 * Inputs:
 * - `value`: the input.
 * - `slope`: the slope for negative inputs.
 * Outputs: the activated value.
 */
pub(crate) fn leaky(value: f32, slope: f32) -> f32 {
    if value > 0.0 { value } else { slope * value }
}

/**
 * Description: The hidden activation on one pre-activation: Leaky ReLU, squared when
 * `square_activation` is on.
 * Inputs:
 * - `head`: the activation settings.
 * - `pre`: one pre-activation.
 * Outputs: the activated value.
 */
pub(crate) fn activate(head: &ClassifierHead, pre: f32) -> f32 {
    let a = leaky(pre, head.leaky_slope);
    if head.square_activation { a * a } else { a }
}

/**
 * Description: Runs the first layer and its activation.
 * Inputs:
 * - `head`: the weights.
 * - `features`: one feature vector, `feature_dim` long.
 * Outputs: the hidden activations, `hidden_dim` long.
 */
pub(crate) fn hidden(head: &ClassifierHead, features: &[f32]) -> Vec<f32> {
    pre_activations(head, features)
        .into_iter()
        .map(|pre| activate(head, pre))
        .collect()
}

/**
 * Description: Runs the head on one feature vector: the hidden layer, then `W2 h + b2`, then the
 * softcap `c * tanh(z / c)` when `logit_softcap` is above 0.
 * Inputs:
 * - `head`: the weights.
 * - `features`: one feature vector, `feature_dim` long.
 * Outputs: one score (logit) per emotion, `num_classes` long.
 */
pub(crate) fn forward(head: &ClassifierHead, features: &[f32]) -> Vec<f32> {
    let h   = hidden(head, features);
    let cap = head.logit_softcap;
    (0..head.num_classes)
        .map(|k| {
            let row = &head.w2[k * head.hidden_dim..(k + 1) * head.hidden_dim];
            let z   = head.b2[k] + row.iter().zip(&h).map(|(w, x)| w * x).sum::<f32>();
            if cap > 0.0 { cap * (z / cap).tanh() } else { z }
        })
        .collect()
}
