//! Step 3.2: the head's weights and its forward pass from features to emotion scores.
//!
//! APIs:
//! - `ClassifierHead`: the head's weights, two linear layers
//! - `init_weights`: creates the head with random starting weights
//! - `hidden`: the first layer's output after Leaky ReLU
//! - `forward`: runs the head on one feature vector

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

use crate::Hyperparameters;

/**
 * Description: The classification head's weights, two linear layers from features to emotions.
 * Each weight matrix is stored row by row in one flat vector: row `j` of `w1` is
 * `w1[j * feature_dim .. (j + 1) * feature_dim]`.
 * Fields:
 * - `feature_dim`, `hidden_dim`, `num_classes`: the layer sizes.
 * - `leaky_slope`: the hidden layer's slope for negative inputs; 0 makes it a plain ReLU.
 * - `w1`, `b1`: first layer, `feature_dim` to `hidden_dim`.
 * - `w2`, `b2`: second layer, `hidden_dim` to `num_classes`.
 */
#[derive(Clone, Serialize)]
pub(crate) struct ClassifierHead {
    pub(crate) feature_dim: usize,
    pub(crate) hidden_dim:  usize,
    pub(crate) num_classes: usize,
    pub(crate) leaky_slope: f32,
    pub(crate) w1:          Vec<f32>,
    pub(crate) b1:          Vec<f32>,
    pub(crate) w2:          Vec<f32>,
    pub(crate) b2:          Vec<f32>,
}

/**
 * Description: Creates the head with He-uniform random weights and zero biases, seeded by
 * `hp.seed` so every run starts from the same weights.
 * Inputs:
 * - `hp`: the sizes to build and the seed.
 * Outputs: the new head.
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
        hidden_dim:  hp.hidden_dim,
        num_classes: hp.num_classes,
        leaky_slope: hp.leaky_slope,
        w1,
        b1: vec![0.0; hp.hidden_dim],
        w2,
        b2: vec![0.0; hp.num_classes],
    }
}

/**
 * Description: Runs the first layer and its Leaky ReLU: `h = W1 x + b1` where positive, and
 * `leaky_slope` times it where negative.
 * Inputs:
 * - `head`: the weights.
 * - `features`: one feature vector, `feature_dim` long.
 * Outputs: the hidden activations, `hidden_dim` long.
 */
pub(crate) fn hidden(head: &ClassifierHead, features: &[f32]) -> Vec<f32> {
    (0..head.hidden_dim)
        .map(|j| {
            let row = &head.w1[j * head.feature_dim..(j + 1) * head.feature_dim];
            let sum = head.b1[j] + row.iter().zip(features).map(|(w, x)| w * x).sum::<f32>();
            if sum > 0.0 { sum } else { head.leaky_slope * sum }
        })
        .collect()
}

/**
 * Description: Runs the head on one feature vector: the hidden layer, then `W2 h + b2`.
 * Inputs:
 * - `head`: the weights.
 * - `features`: one feature vector, `feature_dim` long.
 * Outputs: one score (logit) per emotion, `num_classes` long.
 */
pub(crate) fn forward(head: &ClassifierHead, features: &[f32]) -> Vec<f32> {
    let h = hidden(head, features);
    (0..head.num_classes)
        .map(|k| {
            let row = &head.w2[k * head.hidden_dim..(k + 1) * head.hidden_dim];
            head.b2[k] + row.iter().zip(&h).map(|(w, x)| w * x).sum::<f32>()
        })
        .collect()
}
