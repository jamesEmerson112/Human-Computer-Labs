# Emotion wheel gestures

This catalog transcribes all three rings of the emotion wheel supplied in chat:
**6 core emotions, 36 middle-ring entries, and 72 outer-ring entries**.
There are 114 separately playable gestures and 110 distinct emotion words.
Repeated words (Abandoned, Inadequate, Inferior, and Insecure) retain their
branch context and have separate movements. The original nine gestures are
also retained, for **123 gestures total**.

In the viewer, open **Try a movement > Gestures**, expand an emotion family,
choose an **Emotion**, and click **Play emotion**. Start from **Reset pose**
at Normal pace when comparing expressions. **Stop motion** interrupts any clip.
Click **Reload gestures** after editing a JSON file; restart an older running
viewer once to load the new grouped controls.

These are authored motion interpretations, not validated universal emotional
signals. Five joints cannot make every nearby word unambiguously recognizable.
Each entry has a distinct trajectory, including its pose, accents, pauses,
and timing. All movements return to the starting pose and use the existing
position and speed constraints. Near a limit, offsets may reverse per joint;
this preserves movement but changes the intended expression. Physical tracking
may lag or overshoot. Lighting remains under the existing manual controls.

The pose offsets account for the supplied lamp facing toward base -x at zero.
They are modest relative movements rather than jumps to absolute poses.
Durations are playback seconds: Normal pace takes twice the specified time.

## Complete wheel

### Anger

Core gesture: [Anger](emotion_anger.json). A squared stance with two firm downward accents.

| Middle ring | Outer ring |
|---|---|
| [Hurt](emotion_anger_hurt.json) | [Embarrassed](emotion_anger_hurt_embarrassed.json), [Devastated](emotion_anger_hurt_devastated.json) |
| [Threatened](emotion_anger_threatened.json) | [Insecure](emotion_anger_threatened_insecure.json), [Jealous](emotion_anger_threatened_jealous.json) |
| [Hateful](emotion_anger_hateful.json) | [Resentful](emotion_anger_hateful_resentful.json), [Violated](emotion_anger_hateful_violated.json) |
| [Mad](emotion_anger_mad.json) | [Furious](emotion_anger_mad_furious.json), [Enraged](emotion_anger_mad_enraged.json) |
| [Aggressive](emotion_anger_aggressive.json) | [Provoked](emotion_anger_aggressive_provoked.json), [Hostile](emotion_anger_aggressive_hostile.json) |
| [Frustrated](emotion_anger_frustrated.json) | [Infuriated](emotion_anger_frustrated_infuriated.json), [Irritated](emotion_anger_frustrated_irritated.json) |
| [Distant](emotion_anger_distant.json) | [Withdrawn](emotion_anger_distant_withdrawn.json), [Suspicious](emotion_anger_distant_suspicious.json) |
| [Critical](emotion_anger_critical.json) | [Skeptical](emotion_anger_critical_skeptical.json), [Sarcastic](emotion_anger_critical_sarcastic.json) |

### Disgust

Core gesture: [Disgust](emotion_disgust.json). Recoil slightly and turn the shade away.

| Middle ring | Outer ring |
|---|---|
| [Disapproval](emotion_disgust_disapproval.json) | [Judgmental](emotion_disgust_disapproval_judgmental.json), [Loathing](emotion_disgust_disapproval_loathing.json) |
| [Disappointed](emotion_disgust_disappointed.json) | [Repugnant](emotion_disgust_disappointed_repugnant.json), [Revolted](emotion_disgust_disappointed_revolted.json) |
| [Awful](emotion_disgust_awful.json) | [Revulsion](emotion_disgust_awful_revulsion.json), [Detestable](emotion_disgust_awful_detestable.json) |
| [Avoidance](emotion_disgust_avoidance.json) | [Aversion](emotion_disgust_avoidance_aversion.json), [Hesitant](emotion_disgust_avoidance_hesitant.json) |

### Sad

Core gesture: [Sad](emotion_sad.json). Lower the arm and shade gradually, then recover slowly.

| Middle ring | Outer ring |
|---|---|
| [Guilty](emotion_sad_guilty.json) | [Remorseful](emotion_sad_guilty_remorseful.json), [Ashamed](emotion_sad_guilty_ashamed.json) |
| [Abandoned](emotion_sad_abandoned.json) | [Ignored](emotion_sad_abandoned_ignored.json), [Victimized](emotion_sad_abandoned_victimized.json) |
| [Despair](emotion_sad_despair.json) | [Powerless](emotion_sad_despair_powerless.json), [Vulnerable](emotion_sad_despair_vulnerable.json) |
| [Depressed](emotion_sad_depressed.json) | [Inferior](emotion_sad_depressed_inferior.json), [Empty](emotion_sad_depressed_empty.json) |
| [Lonely](emotion_sad_lonely.json) | [Abandoned](emotion_sad_lonely_abandoned.json), [Isolated](emotion_sad_lonely_isolated.json) |
| [Bored](emotion_sad_bored.json) | [Apathetic](emotion_sad_bored_apathetic.json), [Indifferent](emotion_sad_bored_indifferent.json) |

### Happy

Core gesture: [Happy](emotion_happy.json). Lift the shade and make two welcoming bobs.

| Middle ring | Outer ring |
|---|---|
| [Joyful](emotion_happy_joyful.json) | [Liberated](emotion_happy_joyful_liberated.json), [Ecstatic](emotion_happy_joyful_ecstatic.json) |
| [Interested](emotion_happy_interested.json) | [Amused](emotion_happy_interested_amused.json), [Inquisitive](emotion_happy_interested_inquisitive.json) |
| [Proud](emotion_happy_proud.json) | [Important](emotion_happy_proud_important.json), [Confident](emotion_happy_proud_confident.json) |
| [Accepted](emotion_happy_accepted.json) | [Respected](emotion_happy_accepted_respected.json), [Fulfilled](emotion_happy_accepted_fulfilled.json) |
| [Powerful](emotion_happy_powerful.json) | [Courageous](emotion_happy_powerful_courageous.json), [Provocative](emotion_happy_powerful_provocative.json) |
| [Peaceful](emotion_happy_peaceful.json) | [Loving](emotion_happy_peaceful_loving.json), [Hopeful](emotion_happy_peaceful_hopeful.json) |
| [Intimate](emotion_happy_intimate.json) | [Playful](emotion_happy_intimate_playful.json), [Sensitive](emotion_happy_intimate_sensitive.json) |
| [Optimistic](emotion_happy_optimistic.json) | [Open](emotion_happy_optimistic_open.json), [Inspired](emotion_happy_optimistic_inspired.json) |

### Surprise

Core gesture: [Surprise](emotion_surprise.json). Pull back and raise the shade, then freeze briefly.

| Middle ring | Outer ring |
|---|---|
| [Excited](emotion_surprise_excited.json) | [Eager](emotion_surprise_excited_eager.json), [Energetic](emotion_surprise_excited_energetic.json) |
| [Amazed](emotion_surprise_amazed.json) | [Astonished](emotion_surprise_amazed_astonished.json), [Awe](emotion_surprise_amazed_awe.json) |
| [Confused](emotion_surprise_confused.json) | [Disillusioned](emotion_surprise_confused_disillusioned.json), [Perplexed](emotion_surprise_confused_perplexed.json) |
| [Startled](emotion_surprise_startled.json) | [Shocked](emotion_surprise_startled_shocked.json), [Dismayed](emotion_surprise_startled_dismayed.json) |

### Fear

Core gesture: [Fear](emotion_fear.json). Draw the arm inward and check to each side.

| Middle ring | Outer ring |
|---|---|
| [Humiliated](emotion_fear_humiliated.json) | [Ridiculed](emotion_fear_humiliated_ridiculed.json), [Disrespected](emotion_fear_humiliated_disrespected.json) |
| [Rejected](emotion_fear_rejected.json) | [Alienated](emotion_fear_rejected_alienated.json), [Inadequate](emotion_fear_rejected_inadequate.json) |
| [Submissive](emotion_fear_submissive.json) | [Insignificant](emotion_fear_submissive_insignificant.json), [Worthless](emotion_fear_submissive_worthless.json) |
| [Insecure](emotion_fear_insecure.json) | [Inferior](emotion_fear_insecure_inferior.json), [Inadequate](emotion_fear_insecure_inadequate.json) |
| [Anxious](emotion_fear_anxious.json) | [Worried](emotion_fear_anxious_worried.json), [Overwhelmed](emotion_fear_anxious_overwhelmed.json) |
| [Scared](emotion_fear_scared.json) | [Frightened](emotion_fear_scared_frightened.json), [Terrified](emotion_fear_scared_terrified.json) |
