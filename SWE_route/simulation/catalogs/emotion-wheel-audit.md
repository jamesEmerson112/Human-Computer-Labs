# Emotion wheel coverage and migration audit

The user-selected wheel excludes the entire **Bad** branch. This is an authored motion proposal, not a recognition study or a record of human passes.

The target has **117 emotion entries** (6 core, 37 middle, 74 outer), 114 distinct words, and nine unchanged utility gestures: **126 clips**. Neutral / Reset pose remains a separate control.

The original 114 emotion entries yield 25 exact paths, 40 family-name-only renames, 29 explicitly reassigned motions, and 23 newly authored motions. The latter two categories total 52 targets requiring fresh semantic review.

Migration maps 103 old IDs one-to-one, adds 23 IDs without an old equivalent, and archives 20 unmapped old IDs. Only 74 mappings are eligible to preserve a pass, and only with verified identity and target revision.

## Frozen evidence and review rules

- [Archive manifest](../gestures/archive/six-core-wheel/manifest.json) hashes the byte-preserved 123 original JSON files and original catalog documents.
- [Machine-readable audit](emotion-wheel-audit.json) records every full path, source and target file hash, semantic decision, and reason.
- [Migration manifest](emotion-wheel-migration.json) separates actual mappings, additions, and retired IDs. Matching a leaf word alone is not sufficient.
- Checked means pass; every unchecked clip is failed. New or contextually reassigned clips start unchecked. Existing notes and prior approvals remain in review history.
- Motion steps are preserved exactly for all reused clips. Only new authored clips introduce keyframes. Later user-reviewed failures may be rewritten separately.
- Human review progresses through the six cores, 37 middle entries, then 74 leaves at consistent starting pose, camera, and pace. Technical checks cannot establish emotional recognizability.

## Duplicate words and exclusions

Inferior appears under Fearful / Insecure and Sad / Depressed. Embarrassed appears under Disgusted / Disapproving and Sad / Hurt. Disappointed appears as a Disgusted branch and a Sad / Hurt leaf. Each full path has its own ID and motion.

Removing Bad removes its 13 paths, including Tired, Stressed, Busy, and Bored and their leaves. Fearful / Anxious / Overwhelmed stays: exclusion is by full branch, not by word.

## Motion review boundaries

The lamp has five joints. Motion descriptions are authored hypotheses and may be confused with nearby feelings. Small differences, duplicate trajectories, and near matches are review candidates, not automatic semantic failures. At joint limits, reverse offsets can alter the intended expression; actual MuJoCo motion may lag or overshoot the bounded command.

Exact step-array duplicates in this candidate emotion library: **0 groups**. The separate trajectory audit measures neutral/lower/upper starts, stopping, requested motion, physical tracking, and near similarities.

## Complete mapping

| Target path | Decision | Original path | Pass eligible | Reason |
|---|---|---|---|---|
| Fearful | family_rename | Fear | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Scared | family_rename | Fear / Scared | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Scared / Helpless | authored | New motion | No | Helpless emphasizes blocked action, unlike the retained Frightened double flinch. |
| Fearful / Scared / Frightened | family_rename | Fear / Scared / Frightened | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Anxious | family_rename | Fear / Anxious | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Anxious / Overwhelmed | family_rename | Fear / Anxious / Overwhelmed | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Anxious / Worried | family_rename | Fear / Anxious / Worried | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Insecure | family_rename | Fear / Insecure | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Insecure / Inadequate | family_rename | Fear / Insecure / Inadequate | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Insecure / Inferior | family_rename | Fear / Insecure / Inferior | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Weak | contextual_reuse | Fear / Submissive | No | A small sustained bow can suggest reduced agency; the new Weak meaning needs review. |
| Fearful / Weak / Worthless | contextual_reuse | Fear / Submissive / Worthless | No | Retain the slow contraction for the same leaf word, now under Weak. |
| Fearful / Weak / Insignificant | contextual_reuse | Fear / Submissive / Insignificant | No | Retain the two-stage contraction for the same leaf word, now under Weak. |
| Fearful / Rejected | family_rename | Fear / Rejected | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Fearful / Rejected / Excluded | contextual_reuse | Fear / Rejected / Alienated | No | A withdrawn side-facing stance is a candidate for exclusion; Alienated is not an exact synonym. |
| Fearful / Rejected / Persecuted | authored | New motion | No | Persecuted suggests being pursued; repeated rearward checks differ from Excluded's held withdrawal. |
| Fearful / Threatened | contextual_reuse | Anger / Threatened | No | The old braced retreat is relevant to threat but moves from Anger to Fearful. |
| Fearful / Threatened / Nervous | authored | New motion | No | Nervous uses small uneven checking turns rather than the larger sustained exposure retreat. |
| Fearful / Threatened / Exposed | authored | New motion | No | Exposed begins in visible openness before a protective withdrawal, unlike a repeated nervous scan. |
| Angry | family_rename | Anger | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Let down | authored | New motion | No | Let down needs failed expectation; the original Hurt motion is reserved for Sad / Hurt. |
| Angry / Let down / Betrayed | authored | New motion | No | Betrayed juxtaposes initial openness and a sudden guarded look back; Resentful stays withdrawn. |
| Angry / Let down / Resentful | contextual_reuse | Anger / Hateful / Resentful | No | Keep the low sideways withdrawal; Resentful now belongs under Let down. |
| Angry / Humiliated | contextual_reuse | Fear / Humiliated | No | Retain the low averted posture for Humiliated, now in the Angry family. |
| Angry / Humiliated / Disrespected | contextual_reuse | Fear / Humiliated / Disrespected | No | Retain the interrupted approach for Disrespected, now in the Angry family. |
| Angry / Humiliated / Ridiculed | contextual_reuse | Fear / Humiliated / Ridiculed | No | Retain the double flinch for Ridiculed, now in the Angry family. |
| Angry / Bitter | contextual_reuse | Anger / Hateful | No | The deliberate refusing sweep is a candidate for lingering bitterness; meaning changed. |
| Angry / Bitter / Indignant | authored | New motion | No | Indignant expresses a firm objection with an upright head rather than Violated's protective contraction. |
| Angry / Bitter / Violated | contextual_reuse | Anger / Hateful / Violated | No | Retain the protective contraction for Violated, now under Bitter. |
| Angry / Mad | family_rename | Anger / Mad | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Mad / Furious | family_rename | Anger / Mad / Furious | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Mad / Jealous | contextual_reuse | Anger / Threatened / Jealous | No | Retain the fixed sideways watch for Jealous, now under Mad. |
| Angry / Aggressive | family_rename | Anger / Aggressive | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Aggressive / Provoked | family_rename | Anger / Aggressive / Provoked | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Aggressive / Hostile | family_rename | Anger / Aggressive / Hostile | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Frustrated | family_rename | Anger / Frustrated | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Frustrated / Infuriated | family_rename | Anger / Frustrated / Infuriated | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Frustrated / Annoyed | contextual_reuse | Anger / Frustrated / Irritated | No | Two small neck flicks fit a candidate annoyance cue; the label changes from Irritated. |
| Angry / Distant | family_rename | Anger / Distant | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Distant / Withdrawn | family_rename | Anger / Distant / Withdrawn | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Distant / Numb | authored | New motion | No | Numb deliberately has low movement and a long still interval; it does not use Suspicious checking behavior. |
| Angry / Critical | family_rename | Anger / Critical | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Critical / Skeptical | family_rename | Anger / Critical / Skeptical | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Angry / Critical / Dismissive | authored | New motion | No | Dismissive ends attention decisively, unlike Skeptical's continuing sideways inspection. |
| Disgusted | family_rename | Disgust | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Disgusted / Disapproving | contextual_reuse | Disgust / Disapproval | No | Disapproval changes grammatical form to Disapproving; conservative review reset for this branch rename. |
| Disgusted / Disapproving / Judgmental | contextual_reuse | Disgust / Disapproval / Judgmental | No | Retain the measured inspection; conservative review reset for the Disapproval branch rename. |
| Disgusted / Disapproving / Embarrassed | authored | New motion | No | This Embarrassed branch depicts discomfort at what is observed; Sad / Hurt / Embarrassed keeps its inward hesitant bow. |
| Disgusted / Disappointed | family_rename | Disgust / Disappointed | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Disgusted / Disappointed / Appalled | authored | New motion | No | Appalled combines disbelief and rejection; the retained Revolted clip has a repeated shudder. |
| Disgusted / Disappointed / Revolted | family_rename | Disgust / Disappointed / Revolted | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Disgusted / Awful | family_rename | Disgust / Awful | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Disgusted / Awful / Nauseated | authored | New motion | No | Nauseated uses a slow uncomfortable sway, not the old Revulsion clip's sharp tremors. |
| Disgusted / Awful / Detestable | family_rename | Disgust / Awful / Detestable | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Disgusted / Repelled | contextual_reuse | Disgust / Avoidance | No | A sustained averted stance is a candidate for repulsion; Avoidance described an action. |
| Disgusted / Repelled / Horrified | authored | New motion | No | Horrified has a large fast protective retreat and sustained avoidance; Hesitant makes an interrupted approach. |
| Disgusted / Repelled / Hesitant | contextual_reuse | Disgust / Avoidance / Hesitant | No | Retain the interrupted approach for Hesitant, now under Repelled. |
| Sad | exact | Sad | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Hurt | contextual_reuse | Anger / Hurt | No | The flinch followed by a droop is retained for Hurt, moving from Anger to Sad. |
| Sad / Hurt / Embarrassed | contextual_reuse | Anger / Hurt / Embarrassed | No | Retain the hesitant averted dip for Embarrassed, moving from Anger to Sad. |
| Sad / Hurt / Disappointed | authored | New motion | No | Disappointed in Hurt emphasizes unreturned expectation; Disgusted / Disappointed retains its distinct older movement. |
| Sad / Depressed | exact | Sad / Depressed | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Depressed / Inferior | exact | Sad / Depressed / Inferior | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Depressed / Empty | exact | Sad / Depressed / Empty | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Guilty | exact | Sad / Guilty | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Guilty / Remorseful | exact | Sad / Guilty / Remorseful | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Guilty / Ashamed | exact | Sad / Guilty / Ashamed | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Despair | exact | Sad / Despair | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Despair / Powerless | exact | Sad / Despair / Powerless | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Despair / Grief | contextual_reuse | Anger / Hurt / Devastated | No | A heavy two-stage fold is a candidate for grief; Devastated is not treated as an equivalent label. |
| Sad / Vulnerable | contextual_reuse | Sad / Despair / Vulnerable | No | The tentative retreat moves from an outer leaf to its own Vulnerable branch. |
| Sad / Vulnerable / Fragile | authored | New motion | No | Fragile uses an interrupted low-amplitude lift; Victimized holds a stronger defensive withdrawal. |
| Sad / Vulnerable / Victimized | contextual_reuse | Sad / Abandoned / Victimized | No | Retain the protective contraction for Victimized, now under Vulnerable. |
| Sad / Lonely | exact | Sad / Lonely | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Lonely / Abandoned | exact | Sad / Lonely / Abandoned | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Sad / Lonely / Isolated | exact | Sad / Lonely / Isolated | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy | exact | Happy | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Optimistic | exact | Happy / Optimistic | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Optimistic / Inspired | exact | Happy / Optimistic / Inspired | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Optimistic / Hopeful | contextual_reuse | Happy / Peaceful / Hopeful | No | Retain the gradual upward gaze for Hopeful, now under Optimistic. |
| Happy / Trusting | authored | New motion | No | Trusting is a gradual commitment to approach; Intimate stays close and Sensitive uses smaller tentative movement. |
| Happy / Trusting / Intimate | contextual_reuse | Happy / Intimate | No | A close lingering approach becomes the Intimate leaf under Trusting, rather than a middle-ring category. |
| Happy / Trusting / Sensitive | contextual_reuse | Happy / Intimate / Sensitive | No | Retain the careful approach for Sensitive, now under Trusting. |
| Happy / Peaceful | exact | Happy / Peaceful | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Peaceful / Thankful | authored | New motion | No | Thankful centers a deliberate bow and recovery, while Loving remains close and softly dips. |
| Happy / Peaceful / Loving | exact | Happy / Peaceful / Loving | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Powerful | exact | Happy / Powerful | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Powerful / Creative | authored | New motion | No | Creative uses exploration followed by commitment, unlike Courageous's retreat then direct advance. |
| Happy / Powerful / Courageous | exact | Happy / Powerful / Courageous | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Accepted | exact | Happy / Accepted | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Accepted / Valued | contextual_reuse | Happy / Accepted / Fulfilled | No | The relaxed open pose is a candidate for feeling valued; Fulfilled expressed a different meaning. |
| Happy / Accepted / Respected | exact | Happy / Accepted / Respected | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Proud | exact | Happy / Proud | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Proud / Confident | exact | Happy / Proud / Confident | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Proud / Successful | authored | New motion | No | Successful shows effort resolving into achievement; Confident starts with steady assured movement. |
| Happy / Interested | exact | Happy / Interested | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Interested / Inquisitive | exact | Happy / Interested / Inquisitive | Yes | Full path, meaning, and motion retained; active line endings normalized to LF. |
| Happy / Interested / Curious | authored | New motion | No | Curious focuses on one object with a second approach; Inquisitive's retained clip checks two viewing angles. |
| Happy / Content | authored | New motion | No | Content is restrained and unhurried; Joyful and Free are its more active distinct leaves. |
| Happy / Content / Joyful | contextual_reuse | Happy / Joyful | No | Retain the welcoming double bounce for Joyful as a leaf under Content. |
| Happy / Content / Free | contextual_reuse | Happy / Joyful / Liberated | No | The broad unfolding is a candidate for feeling free, moved from Liberated under Joyful. |
| Happy / Playful | contextual_reuse | Happy / Intimate / Playful | No | The peek and quick bob become a middle-ring Playful category rather than a leaf under Intimate. |
| Happy / Playful / Cheeky | authored | New motion | No | Cheeky emphasizes a mismatched body and head direction followed by a teasing beat; no arbitrary jitter. |
| Happy / Playful / Aroused | authored | New motion | No | Aroused is interpreted as heightened activation and attention, expressed by a wake-and-orient sequence. |
| Surprised | family_rename | Surprise | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Excited | family_rename | Surprise / Excited | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Excited / Energetic | family_rename | Surprise / Excited / Energetic | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Excited / Eager | family_rename | Surprise / Excited / Eager | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Amazed | family_rename | Surprise / Amazed | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Amazed / Awe | family_rename | Surprise / Amazed / Awe | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Amazed / Astonished | family_rename | Surprise / Amazed / Astonished | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Confused | family_rename | Surprise / Confused | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Confused / Perplexed | family_rename | Surprise / Confused / Perplexed | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Confused / Disillusioned | family_rename | Surprise / Confused / Disillusioned | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Startled | family_rename | Surprise / Startled | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Startled / Dismayed | family_rename | Surprise / Startled / Dismayed | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |
| Surprised / Startled / Shocked | family_rename | Surprise / Startled / Shocked | Yes | Only the core-family name changes; branch meaning and all motion steps are retained. |

## Unmapped original paths

These files remain in the frozen archive; they are no longer in the active catalog.

- Anger / Critical / Sarcastic (`emotion_anger_critical_sarcastic`)
- Anger / Distant / Suspicious (`emotion_anger_distant_suspicious`)
- Anger / Mad / Enraged (`emotion_anger_mad_enraged`)
- Anger / Threatened / Insecure (`emotion_anger_threatened_insecure`)
- Disgust / Avoidance / Aversion (`emotion_disgust_avoidance_aversion`)
- Disgust / Awful / Revulsion (`emotion_disgust_awful_revulsion`)
- Disgust / Disappointed / Repugnant (`emotion_disgust_disappointed_repugnant`)
- Disgust / Disapproval / Loathing (`emotion_disgust_disapproval_loathing`)
- Fear / Rejected / Inadequate (`emotion_fear_rejected_inadequate`)
- Fear / Scared / Terrified (`emotion_fear_scared_terrified`)
- Happy / Interested / Amused (`emotion_happy_interested_amused`)
- Happy / Joyful / Ecstatic (`emotion_happy_joyful_ecstatic`)
- Happy / Optimistic / Open (`emotion_happy_optimistic_open`)
- Happy / Powerful / Provocative (`emotion_happy_powerful_provocative`)
- Happy / Proud / Important (`emotion_happy_proud_important`)
- Sad / Abandoned (`emotion_sad_abandoned`)
- Sad / Abandoned / Ignored (`emotion_sad_abandoned_ignored`)
- Sad / Bored (`emotion_sad_bored`)
- Sad / Bored / Apathetic (`emotion_sad_bored_apathetic`)
- Sad / Bored / Indifferent (`emotion_sad_bored_indifferent`)
