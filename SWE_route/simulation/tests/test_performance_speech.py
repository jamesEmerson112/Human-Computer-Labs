"""Check that portable speech fixtures preserve text and measured audio timing."""

from array import array
import hashlib
import json
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1] / "performances"
MANIFEST = json.loads((ROOT / "demo.json").read_text(encoding="utf-8"))
SCENES = MANIFEST["scenes"]
PHRASES = [phrase for scene in SCENES for phrase in scene["phrases"]]


def test_demo_preserves_exact_input_text_and_phrase_order():
    assert MANIFEST["schema_version"] == 1
    expected = {
        "greeting": "Oh! You're back. I missed you.",
        "question": "What's that? Can I take a closer look?",
        "reassurance": "It's okay. We can figure this out together.",
    }
    assert len(SCENES) == len(expected)
    assert {scene["id"] for scene in SCENES} == set(expected)
    for scene in SCENES:
        assert scene["text"] == expected[scene["id"]]
        assert " ".join(phrase["text"] for phrase in scene["phrases"]) == scene["text"]
        assert len(scene["phrases"]) >= 2


def test_manifest_covers_only_local_audio_without_orphan_files():
    paths = []
    for phrase in PHRASES:
        path = (ROOT / phrase["audio"]).resolve()
        assert path.is_relative_to((ROOT / "audio").resolve())
        assert path.suffix == ".wav"
        paths.append(path)
    assert len(paths) == len(set(paths)) == 7
    assert set(paths) == {path.resolve() for path in (ROOT / "audio").glob("*.wav")}
    assert sum(path.stat().st_size for path in paths) < 1_000_000


@pytest.mark.parametrize("phrase", PHRASES, ids=[phrase["audio"] for phrase in PHRASES])
def test_audio_is_audible_pcm_with_exact_manifest_duration_and_digest(phrase: dict):
    path = ROOT / phrase["audio"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == phrase["sha256"]
    with wave.open(str(path), "rb") as audio:
        assert audio.getcomptype() == "NONE"
        assert audio.getnchannels() == MANIFEST["voice"]["channels"] == 1
        assert audio.getsampwidth() == MANIFEST["voice"]["sample_width_bytes"] == 2
        assert audio.getframerate() == MANIFEST["voice"]["sample_rate_hz"] == 22050
        assert audio.getnframes() == phrase["frames"]
        assert phrase["duration_s"] == pytest.approx(
            audio.getnframes() / audio.getframerate(), abs=1e-12
        )
        assert 0.25 < phrase["duration_s"] < 10
        pcm = audio.readframes(audio.getnframes())
        assert len(pcm) == audio.getnframes() * 2
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    # Catch empty or effectively silent fixture output without judging pronunciation.
    assert max(abs(sample) for sample in samples) > 1000
    assert sum(abs(sample) > 256 for sample in samples) / 22050 > 0.1
