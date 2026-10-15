# Animation motion audit

126 animations; 117 emotions; 756 controller cases.
Command failures: 0. Reset failures: 0.

Controller checks cover neutral and both near-limit starting poses at Normal and Full pace, completion, requested limits/speed, and interruption. Emotion clips must move and return to their starting pose. Utility clips have their original hold/reach behavior.

## Similarity review candidates

0 identical step groups and 10 close trajectory pairs. These flags do not assign human pass/fail.

The comparison resamples neutral requested trajectories to 128 phase samples. A pair is flagged at RMS difference <=2 degrees, maximum difference <=6 degrees, and duration ratio <=1.25. It can miss perceptual similarity and timing distinctions. Full nearest-neighbor results are in audit.json.

| First animation | Second animation | RMS difference (degrees) | Duration ratio |
|---|---|---:|---:|
| Sad / Despair | Sad / Despair / Grief | 0.344 | 1.110 |
| Fearful / Insecure / Inadequate | Surprised / Confused / Disillusioned | 1.266 | 1.167 |
| Happy / Peaceful / Loving | Happy / Trusting / Intimate | 1.377 | 1.066 |
| Fearful / Weak / Worthless | Sad / Despair | 1.426 | 1.025 |
| Fearful / Weak / Worthless | Sad / Despair / Grief | 1.466 | 1.138 |
| Fearful / Weak / Worthless | Sad / Depressed | 1.541 | 1.124 |
| Angry / Humiliated | Sad / Guilty / Ashamed | 1.555 | 1.103 |
| Sad / Depressed | Sad / Despair | 1.814 | 1.153 |
| Disgusted / Disappointed | Sad | 1.860 | 1.171 |
| Fearful / Rejected / Excluded | Sad / Lonely / Isolated | 1.866 | 1.209 |

## Actual physics observations

Hash-matched reference: `benchmarks\2026-10-14-wheel-audit\reference`. 126 neutral-start default-MuJoCo trials.

Maximum actual soft-limit overshoot: 1.027361 degrees. Maximum tracking error: 4.430200 degrees.
Trials with contacts: 0. Trials with solver warnings: 0.

These observations do not establish physical tracking from near-limit starting poses, hardware accuracy, or emotional recognition. Requested-angle checks and actual physics observations are separate.
