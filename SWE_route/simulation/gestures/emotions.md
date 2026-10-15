# Emotion wheel gestures

This catalog follows the user-selected wheel with the entire **Bad** branch excluded: **6 core emotions, 37 middle-ring entries, and 74 outer-ring entries**. The 117 full paths contain 114 distinct words. With nine original utility gestures, the active library has **126 clips**.

The core families are **Fearful, Angry, Disgusted, Sad, Happy, and Surprised**. **Neutral / Reset pose** is a separate action, not a seventh emotion clip. `idle_hold` holds the current starting pose; it is not a neutral reset.

Use the native viewer's gesture library to select a feeling, play it, and mark **Pass** only if its movement works for you. Every unchecked entry counts as failed. Compare from Reset pose with a consistent camera and pace. Export your review to preserve passes and notes outside the browser.

These are authored interpretations, not validated universal emotional signals. Five joints cannot guarantee that nearby feelings are recognizable. All emotion clips return to their starting pose and use radians, bounded control, and reversible offsets near joint limits. Limit reversal can change expression; actual physics may lag or overshoot. Lighting remains under manual control.

The original 123 files are byte-preserved in [the archive](archive/six-core-wheel/manifest.json). The [coverage audit](../catalogs/emotion-wheel-audit.md) explains every retained, renamed, reassigned, authored, and retired path; [migration metadata](../catalogs/emotion-wheel-migration.json) protects review provenance.

## Complete selected wheel

### Fearful

Core gesture: [Fearful](emotion_fearful.json).

| Middle ring | Outer ring |
|---|---|
| [Scared](emotion_fearful_scared.json) | [Helpless](emotion_fearful_scared_helpless.json), [Frightened](emotion_fearful_scared_frightened.json) |
| [Anxious](emotion_fearful_anxious.json) | [Overwhelmed](emotion_fearful_anxious_overwhelmed.json), [Worried](emotion_fearful_anxious_worried.json) |
| [Insecure](emotion_fearful_insecure.json) | [Inadequate](emotion_fearful_insecure_inadequate.json), [Inferior](emotion_fearful_insecure_inferior.json) |
| [Weak](emotion_fearful_weak.json) | [Worthless](emotion_fearful_weak_worthless.json), [Insignificant](emotion_fearful_weak_insignificant.json) |
| [Rejected](emotion_fearful_rejected.json) | [Excluded](emotion_fearful_rejected_excluded.json), [Persecuted](emotion_fearful_rejected_persecuted.json) |
| [Threatened](emotion_fearful_threatened.json) | [Nervous](emotion_fearful_threatened_nervous.json), [Exposed](emotion_fearful_threatened_exposed.json) |

### Angry

Core gesture: [Angry](emotion_angry.json).

| Middle ring | Outer ring |
|---|---|
| [Let down](emotion_angry_let_down.json) | [Betrayed](emotion_angry_let_down_betrayed.json), [Resentful](emotion_angry_let_down_resentful.json) |
| [Humiliated](emotion_angry_humiliated.json) | [Disrespected](emotion_angry_humiliated_disrespected.json), [Ridiculed](emotion_angry_humiliated_ridiculed.json) |
| [Bitter](emotion_angry_bitter.json) | [Indignant](emotion_angry_bitter_indignant.json), [Violated](emotion_angry_bitter_violated.json) |
| [Mad](emotion_angry_mad.json) | [Furious](emotion_angry_mad_furious.json), [Jealous](emotion_angry_mad_jealous.json) |
| [Aggressive](emotion_angry_aggressive.json) | [Provoked](emotion_angry_aggressive_provoked.json), [Hostile](emotion_angry_aggressive_hostile.json) |
| [Frustrated](emotion_angry_frustrated.json) | [Infuriated](emotion_angry_frustrated_infuriated.json), [Annoyed](emotion_angry_frustrated_annoyed.json) |
| [Distant](emotion_angry_distant.json) | [Withdrawn](emotion_angry_distant_withdrawn.json), [Numb](emotion_angry_distant_numb.json) |
| [Critical](emotion_angry_critical.json) | [Skeptical](emotion_angry_critical_skeptical.json), [Dismissive](emotion_angry_critical_dismissive.json) |

### Disgusted

Core gesture: [Disgusted](emotion_disgusted.json).

| Middle ring | Outer ring |
|---|---|
| [Disapproving](emotion_disgusted_disapproving.json) | [Judgmental](emotion_disgusted_disapproving_judgmental.json), [Embarrassed](emotion_disgusted_disapproving_embarrassed.json) |
| [Disappointed](emotion_disgusted_disappointed.json) | [Appalled](emotion_disgusted_disappointed_appalled.json), [Revolted](emotion_disgusted_disappointed_revolted.json) |
| [Awful](emotion_disgusted_awful.json) | [Nauseated](emotion_disgusted_awful_nauseated.json), [Detestable](emotion_disgusted_awful_detestable.json) |
| [Repelled](emotion_disgusted_repelled.json) | [Horrified](emotion_disgusted_repelled_horrified.json), [Hesitant](emotion_disgusted_repelled_hesitant.json) |

### Sad

Core gesture: [Sad](emotion_sad.json).

| Middle ring | Outer ring |
|---|---|
| [Hurt](emotion_sad_hurt.json) | [Embarrassed](emotion_sad_hurt_embarrassed.json), [Disappointed](emotion_sad_hurt_disappointed.json) |
| [Depressed](emotion_sad_depressed.json) | [Inferior](emotion_sad_depressed_inferior.json), [Empty](emotion_sad_depressed_empty.json) |
| [Guilty](emotion_sad_guilty.json) | [Remorseful](emotion_sad_guilty_remorseful.json), [Ashamed](emotion_sad_guilty_ashamed.json) |
| [Despair](emotion_sad_despair.json) | [Powerless](emotion_sad_despair_powerless.json), [Grief](emotion_sad_despair_grief.json) |
| [Vulnerable](emotion_sad_vulnerable.json) | [Fragile](emotion_sad_vulnerable_fragile.json), [Victimized](emotion_sad_vulnerable_victimized.json) |
| [Lonely](emotion_sad_lonely.json) | [Abandoned](emotion_sad_lonely_abandoned.json), [Isolated](emotion_sad_lonely_isolated.json) |

### Happy

Core gesture: [Happy](emotion_happy.json).

| Middle ring | Outer ring |
|---|---|
| [Optimistic](emotion_happy_optimistic.json) | [Inspired](emotion_happy_optimistic_inspired.json), [Hopeful](emotion_happy_optimistic_hopeful.json) |
| [Trusting](emotion_happy_trusting.json) | [Intimate](emotion_happy_trusting_intimate.json), [Sensitive](emotion_happy_trusting_sensitive.json) |
| [Peaceful](emotion_happy_peaceful.json) | [Thankful](emotion_happy_peaceful_thankful.json), [Loving](emotion_happy_peaceful_loving.json) |
| [Powerful](emotion_happy_powerful.json) | [Creative](emotion_happy_powerful_creative.json), [Courageous](emotion_happy_powerful_courageous.json) |
| [Accepted](emotion_happy_accepted.json) | [Valued](emotion_happy_accepted_valued.json), [Respected](emotion_happy_accepted_respected.json) |
| [Proud](emotion_happy_proud.json) | [Confident](emotion_happy_proud_confident.json), [Successful](emotion_happy_proud_successful.json) |
| [Interested](emotion_happy_interested.json) | [Inquisitive](emotion_happy_interested_inquisitive.json), [Curious](emotion_happy_interested_curious.json) |
| [Content](emotion_happy_content.json) | [Joyful](emotion_happy_content_joyful.json), [Free](emotion_happy_content_free.json) |
| [Playful](emotion_happy_playful.json) | [Cheeky](emotion_happy_playful_cheeky.json), [Aroused](emotion_happy_playful_aroused.json) |

### Surprised

Core gesture: [Surprised](emotion_surprised.json).

| Middle ring | Outer ring |
|---|---|
| [Excited](emotion_surprised_excited.json) | [Energetic](emotion_surprised_excited_energetic.json), [Eager](emotion_surprised_excited_eager.json) |
| [Amazed](emotion_surprised_amazed.json) | [Awe](emotion_surprised_amazed_awe.json), [Astonished](emotion_surprised_amazed_astonished.json) |
| [Confused](emotion_surprised_confused.json) | [Perplexed](emotion_surprised_confused_perplexed.json), [Disillusioned](emotion_surprised_confused_disillusioned.json) |
| [Startled](emotion_surprised_startled.json) | [Dismayed](emotion_surprised_startled_dismayed.json), [Shocked](emotion_surprised_startled_shocked.json) |
