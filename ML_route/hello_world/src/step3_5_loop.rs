//! Step 3.5: batching for the training loop, whose body lives in `main`.
//!
//! APIs:
//! - `next_batch`: picks the next training batch from a fresh shuffle each epoch

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::step3_1_load::Utterance;

/**
 * Description: Picks the next training batch, like `DistributedTokenLoader::next_batch`. The lines
 * are shuffled once per pass through the data (epoch), with a seed made from `seed` and the epoch,
 * so every run sees the same order and each epoch sees a different one.
 * Inputs:
 * - `data`: all training lines.
 * - `batch_size`: lines per batch.
 * - `step`: current step, used to find the epoch and the position inside it.
 * - `seed`: the run's random seed.
 * Outputs: the batch's feature vectors and their labels.
 * Constraints: `data` must not be empty. A batch never spans two epochs, so the last batch of an
 * epoch can be shorter.
 */
pub(crate) fn next_batch(
    data: &[Utterance],
    batch_size: usize,
    step: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let batches_per_epoch = data.len().div_ceil(batch_size);
    let epoch             = step / batches_per_epoch;
    let start             = (step % batches_per_epoch) * batch_size;
    let end               = (start + batch_size).min(data.len());

    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut rng               = StdRng::seed_from_u64(seed.wrapping_add(epoch as u64));
    order.shuffle(&mut rng);

    let picked   = &order[start..end];
    let features = picked.iter().map(|&i| data[i].features.clone()).collect();
    let labels   = picked.iter().map(|&i| data[i].emotion).collect();
    (features, labels)
}
