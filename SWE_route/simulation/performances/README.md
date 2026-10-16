# Local speech fixtures

[`demo.json`](demo.json) contains three hardcoded lines for the first lamp
performance demonstration. Each scene keeps the exact text and lists its phrase
audio files in playback order. The emotion is supplied separately by the viewer;
every emotion uses the same recording so the movement comparison is controlled.

| Scene | Text | Audio duration, excluding interphrase gaps |
| --- | --- | --- |
| `greeting` | Oh! You're back. I missed you. | 4.048073 s |
| `question` | What's that? Can I take a closer look? | 3.803492 s |
| `reassurance` | It's okay. We can figure this out together. | 3.888345 s |

The performance coordinator adds a 0.18-second gap between phrases. Durations in
the manifest are measured from PCM frame counts, including silence present in
the recording. They are phrase timings; no word alignment is provided.

## Provenance and portability

The fixtures were synthesized locally with Microsoft's installed **Microsoft
David Desktop** voice (`TTS_MS_EN-US_DAVID_11.0`, `en-US`) through Windows
`System.Speech.Synthesis.SpeechSynthesizer`, assembly version `4.0.0.0`. Synthesis
used rate `0` and volume `100`, with no emotional voice control or SSML.
`demo.json` records those settings and a SHA-256 digest of each generated file.

All seven recordings are mono, 22,050 Hz, 16-bit PCM WAV, totaling 518,052 bytes.
The repository stores generated audio, not Microsoft's speech engine or voice
package. Regeneration uses the installed Windows voice and its existing terms;
no third-party recording was copied into these fixtures.

Playback is ordinary browser audio and works independently of Windows. There is
no runtime speech service, voice download, speech model, or network dependency.
This demonstration does not synthesize arbitrary text at runtime and does not
claim emotion-dependent speech prosody.

## Regenerate on Windows

From the repository root, using Windows PowerShell with the David Desktop voice
installed:

```powershell
powershell.exe -NoProfile -File SWE_route/simulation/tools/build_performance_speech.ps1
```

The generator writes the seven WAV files and rebuilds the manifest using measured
frame counts and hashes. A different installed voice can be selected explicitly:

```powershell
powershell.exe -NoProfile -File SWE_route/simulation/tools/build_performance_speech.ps1 -VoiceName 'Microsoft Zira Desktop' -Rate 0
```

Voice or Windows updates may change the bytes and durations. Regenerate the
complete set together and rerun validation after any such change. Regeneration
may need normal desktop process access when a sandbox cannot initialize the
Windows voice engine. Runtime playback does not.

From `SWE_route/simulation/`:

```powershell
uv run --locked pytest -q tests/test_performance_speech.py
```

The checks verify exact phrase text, portable PCM format, measured durations,
digests, and audible signal content. They do not judge pronunciation or perceived
voice quality; those remain listening checks in the viewer.
