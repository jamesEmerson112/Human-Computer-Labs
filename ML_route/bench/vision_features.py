"""Finds the speaker's face in each MELD clip's frames and extracts frozen vision features for the head baselines.

Contents:
- ENCODERS, VIEWS: the vision encoders this script runs and the views each one reads
- YUNET_REPO, YUNET_FILE, FACE_SCORE, CROP_SCALE, SPEAKER_RULE: face finding settings and the speaker crop rule
- HSEMOTION_URL: where the HSEmotion EfficientNet-B2 weights are downloaded from
- UNREADABLE, DETECTOR_ERROR: the per-frame face counts that mark a frame YuNet could not search
- pick_dtype: bf16 on GPUs that run it natively, else fp16
- clips_with_frames: the split's lines whose eight frames all exist
- find_faces: runs YuNet on every frame and keeps the largest face, writing FEATS/faces/<split>.json
- face_stats: counts clips and frames with and without faces for faces_stats.json
- speaker_crop: cuts the square speaker crop out of one frame
- load_encoder: loads one encoder and returns its CPU preprocessing and GPU forward functions
- encode_view: encodes one split for one (encoder, view) and returns the rows and per-clip timings
- save_npz: writes one feature file with the same layout as features.save, atomically
- timing: summarises per-clip milliseconds as median and 95th percentile
- update_timing: merges timing entries into RESULTS/vision_timing.json
- main: finds faces, then extracts the chosen encoders and views over the chosen splits
"""

import argparse
import json
import os
import threading
import time
import traceback
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import FEATS, RESULTS, read_split
from frames import N_FRAMES, frame_paths

ENCODERS = {
    "siglip2-base": "google/siglip2-base-patch16-224",
    "vjepa2-vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vit-face-expression": "trpakov/vit-face-expression",
    "hsemotion-enet-b2-8": "HSE-asavchenko/face-emotion-recognition enet_b2_8",
}
"""Description: Vision encoders by short name. HSEmotion is not on Hugging Face; its value only names the
GitHub repo and model file that HSEMOTION_URL points at."""
VIEWS = {
    "siglip2-base": ("frame", "face"),
    "vjepa2-vitl": ("frame",),
    "vit-face-expression": ("face",),
    "hsemotion-enet-b2-8": ("face",),
}
"""Description: The views each encoder reads. "frame" is the whole frame (for V-JEPA 2, the eight frames as one
video) and "face" is the speaker crop."""
YUNET_REPO = "opencv/face_detection_yunet"
"""Description: The Hugging Face repo that holds the YuNet face detector."""
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
"""Description: The YuNet ONNX file inside YUNET_REPO."""
FACE_SCORE = 0.6
"""Description: YuNet detections scoring below this are ignored."""
CROP_SCALE = 1.25
"""Description: The speaker crop's side as a multiple of the face box's longer side, a 25 percent margin."""
SPEAKER_RULE = ("In each frame the largest YuNet face (by box area, score at least 0.6) is taken as the speaker. "
                "The crop is a square centred on that box whose side is 1.25 times the box's longer side, clamped "
                "to the frame, so a face at the frame's edge gives a crop that is no longer square. This is a guess "
                "at the speaker: in two-shots and over-the-shoulder shots the largest face is often the listener.")
"""Description: The speaker crop rule in words, written into faces_stats.json so it travels with the data."""
HSEMOTION_URL = ("https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/models/"
                 "affectnet_emotions/enet_b2_8.pt")
"""Description: The HSEmotion EfficientNet-B2 eight-emotion AffectNet model, a pickled timm module."""
UNREADABLE = -1
"""Description: The face count stored for a frame that cv2.imread could not read."""
DETECTOR_ERROR = -2
"""Description: The face count stored for a frame on which YuNet raised an OpenCV error."""


def pick_dtype(name=None):
    """
    Description: Chooses the encoders' floating point type.
    Inputs:
    - name: "bf16", "fp16" or "fp32" to force one, or None to choose.
    Outputs: a torch dtype; bf16 on GPUs that run it natively (compute capability 8 or above, such as the A40),
    else fp16.
    Constraints: the capability check is used because recent torch reports emulated bf16 as supported on Pascal.
    """
    forced = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name:
        return forced[name]
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def clips_with_frames(split):
    """
    Description: Reads one split's lines and keeps those whose eight frames all exist.
    Inputs:
    - split: "train", "dev" or "test".
    Outputs: (the kept line dicts in label file order, the number of lines skipped for missing frames).
    """
    rows = read_split(split)
    kept = [r for r in rows if all(p.exists() for p in frame_paths(split, r["clip"]))]
    return kept, len(rows) - len(kept)


def find_faces(split, rows, workers=8):
    """
    Description: Runs YuNet on every frame of the given clips and keeps the largest face per frame. Clips
    already in FEATS/faces/<split>.json are not run again; the rest are added and the file is rewritten
    every 1000 clips, so a killed run loses at most that many.
    The number of faces found in each frame goes to FEATS/faces/<split>_counts.json for faces_stats.json, with
    UNREADABLE or DETECTOR_ERROR in place of the count for a frame that could not be searched.
    Inputs:
    - split: "train", "dev" or "test".
    - rows: line dicts whose frames exist.
    - workers: parallel detector threads, each with its own YuNet instance.
    Outputs: (a dict from clip name to N_FRAMES boxes or None, each box [x, y, w, h, score] in frame pixels,
    a list of milliseconds per newly searched clip, the wall-clock seconds the search took).
    Constraints: boxes are stored as detected and may start at negative x or y; speaker_crop clamps them. A
    clip's milliseconds are measured inside one of several threads that share the CPU, so they are slower than
    one clip searched alone.
    """
    import cv2
    from huggingface_hub import hf_hub_download
    folder = FEATS / "faces"
    folder.mkdir(parents=True, exist_ok=True)
    path, count_path = folder / f"{split}.json", folder / f"{split}_counts.json"
    faces = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    counts = json.loads(count_path.read_text(encoding="utf-8")) if count_path.exists() else {}
    todo = [r["clip"] for r in rows if r["clip"] not in faces]
    if not todo:
        return faces, [], 0.0
    onnx = hf_hub_download(YUNET_REPO, YUNET_FILE)
    local = threading.local()
    cv2.setNumThreads(1)

    def one(clip):
        if not hasattr(local, "det"):
            local.det = cv2.FaceDetectorYN.create(onnx, "", (320, 320), FACE_SCORE, 0.3, 5000)
        t = time.perf_counter()
        boxes, found = [], []
        for p in frame_paths(split, clip):
            img = cv2.imread(str(p))
            dets, failed = None, UNREADABLE if img is None else None
            if img is not None:
                try:
                    local.det.setInputSize((img.shape[1], img.shape[0]))
                    _, dets = local.det.detect(img)
                except cv2.error:
                    failed = DETECTOR_ERROR
            dets = [] if dets is None else [d for d in dets if d[14] >= FACE_SCORE]
            found.append(len(dets) if failed is None else failed)
            best = max(dets, key=lambda d: d[2] * d[3]) if dets else None
            boxes.append(None if best is None else [round(float(v), 1) for v in best[:4]] + [round(float(best[14]), 3)])
        return clip, boxes, found, 1000 * (time.perf_counter() - t)

    def checkpoint():
        for target, value in ((path, faces), (count_path, counts)):
            tmp = target.with_suffix(".part")
            tmp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(tmp, target)

    t0, ms = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (clip, boxes, found, took) in enumerate(pool.map(one, todo), 1):
            faces[clip], counts[clip] = boxes, found
            ms.append(took)
            if i % 1000 == 0 or i == len(todo):
                checkpoint()
                print(f"faces {split}: {i} of {len(todo)} clips, {time.perf_counter() - t0:.0f} s", flush=True)
    return faces, ms, time.perf_counter() - t0


def face_stats(split, rows, faces, no_frames):
    """
    Description: Counts clips and frames with and without a face for one split.
    Inputs:
    - split: "train", "dev" or "test".
    - rows: line dicts whose frames exist.
    - faces: the dict from find_faces.
    - no_frames: lines skipped because their frames are missing.
    Outputs: a dict of counts.
    Constraints: prints a warning when frames could not be searched or fewer than half the frames have a face,
    since a detector that fails on the pod would otherwise only show up as all-zero face features.
    """
    count_path = FEATS / "faces" / f"{split}_counts.json"
    counts = json.loads(count_path.read_text(encoding="utf-8")) if count_path.exists() else {}
    boxes = [faces[r["clip"]] for r in rows]
    found = [counts[r["clip"]] for r in rows if r["clip"] in counts]
    stats = {
        "clips": len(rows),
        "clips_without_frames": no_frames,
        "clips_with_no_face": sum(1 for b in boxes if all(x is None for x in b)),
        "frames": len(rows) * N_FRAMES,
        "frames_with_face": sum(1 for b in boxes for x in b if x is not None),
        "frames_with_several_faces": sum(1 for f in found for n in f if n > 1),
        "frames_unreadable": sum(1 for f in found for n in f if n == UNREADABLE),
        "frames_detector_error": sum(1 for f in found for n in f if n == DETECTOR_ERROR),
        "median_face_score": round(float(np.median([x[4] for b in boxes for x in b if x] or [0])), 3),
    }
    if stats["frames_unreadable"] or stats["frames_detector_error"] or stats["frames_with_face"] < stats["frames"] / 2:
        print(f"WARNING faces {split}: {stats['frames_with_face']} of {stats['frames']} frames have a face, "
              f"{stats['frames_unreadable']} unreadable, {stats['frames_detector_error']} detector errors", flush=True)
    return stats


def speaker_crop(img, box):
    """
    Description: Cuts the speaker crop described by SPEAKER_RULE out of one frame.
    Inputs:
    - img: the frame as a PIL image.
    - box: [x, y, w, h, score] in frame pixels.
    Outputs: the crop as a PIL image.
    """
    x, y, w, h = box[:4]
    side = CROP_SCALE * max(w, h)
    cx, cy = x + w / 2, y + h / 2
    left, top = max(0, int(round(cx - side / 2))), max(0, int(round(cy - side / 2)))
    right, bottom = min(img.width, int(round(cx + side / 2))), min(img.height, int(round(cy + side / 2)))
    return img.crop((left, top, max(right, left + 1), max(bottom, top + 1)))


def load_encoder(name, dtype):
    """
    Description: Loads one encoder on the GPU and splits it into a CPU preprocessing function, which encode_view
    runs on several loader threads, and a GPU forward function, which it runs on the main thread. The Hugging
    Face processors resize and normalise on the GPU, because on the CPU they cost 20 to 33 ms per clip, more than
    the A40 needs for the forward pass.
    Inputs:
    - name: a key of ENCODERS.
    - dtype: the torch dtype for weights and inputs.
    Outputs: (prep, forward). prep takes a list of PIL images (for V-JEPA 2, a list of clips, each a list of
    N_FRAMES PIL images) and returns CPU tensors or arrays; forward takes those and returns a float32 array with
    one row per image or clip.
    Constraints: SigLIP 2 gives get_image_features' pooler_output, the attention-pooling head's 768-d image
    embedding that SigLIP compares with text. V-JEPA 2 gives the encoder's last hidden state averaged over all
    its tokens, with the predictor skipped; its processor resizes and centre-crops each frame to 256 x 256, which
    cuts the sides of MELD's wide frames. ViT face expression gives the CLS vector of the last hidden
    state, after the final layer norm. HSEmotion gives the EfficientNet-B2 global max pooled features (1408-d)
    before its classifier, on a 260 x 260 resize with ImageNet normalisation, as the hsemotion package does; its
    batches are padded with blank images to a multiple of 32, which leaves the real rows unchanged in eval mode.
    """
    import transformers
    repo = ENCODERS[name]
    if name in ("siglip2-base", "vit-face-expression"):
        proc = transformers.AutoImageProcessor.from_pretrained(repo, backend="torchvision")
        model = transformers.AutoModel.from_pretrained(repo, dtype=dtype).cuda().eval()

        def prep(images):
            return [torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in images]

        def forward(images):
            px = proc(images=images, device="cuda", return_tensors="pt")["pixel_values"].to(dtype)
            if name == "vit-face-expression":
                return model(pixel_values=px).last_hidden_state[:, 0].float().cpu().numpy()
            out = model.get_image_features(pixel_values=px)
            return (out if torch.is_tensor(out) else out.pooler_output).float().cpu().numpy()
        return prep, forward
    if name == "vjepa2-vitl":
        proc = transformers.AutoVideoProcessor.from_pretrained(repo)
        model = transformers.AutoModel.from_pretrained(repo, dtype=dtype).cuda().eval()

        def prep(videos):
            return [np.stack([np.asarray(f) for f in v]) for v in videos]

        def forward(videos):
            px = proc(videos=videos, device="cuda", return_tensors="pt")["pixel_values_videos"].to(dtype)
            return model(pixel_values_videos=px, skip_predictor=True).last_hidden_state.mean(1).float().cpu().numpy()
        return prep, forward
    if name == "hsemotion-enet-b2-8":
        import timm
        weights = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hsemotion" / "enet_b2_8.pt"
        for attempt in range(4):
            if weights.exists():
                break
            weights.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp = weights.with_suffix(".part")
                tmp.write_bytes(urllib.request.urlopen(HSEMOTION_URL, timeout=120).read())
                os.replace(tmp, weights)
            except OSError:
                # GitHub sometimes answers a large file with a passing 503.
                if attempt == 3:
                    raise
                time.sleep(10)
        # The file is a whole pickled module from an old timm; only its weights and pooling type load into timm 1.x.
        old = torch.load(weights, map_location="cpu", weights_only=False)
        model = timm.create_model("tf_efficientnet_b2", pretrained=False, num_classes=8,
                                  global_pool=old.global_pool.pool_type)
        model.load_state_dict(old.state_dict())
        model.classifier = torch.nn.Identity()
        model = model.to("cuda", dtype).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def prep(images):
            return torch.from_numpy(np.stack([np.asarray(im.convert("RGB").resize((260, 260), resample=2))
                                              for im in images]))

        def forward(arr):
            n = len(arr)
            # Each new batch size cost about 12 s of cuDNN setup in testing, so batches are padded to a multiple of 32.
            arr = torch.cat([arr, arr.new_zeros((-n % 32,) + tuple(arr.shape[1:]))])
            px = arr.cuda().permute(0, 3, 1, 2).float() / 255.0
            return model(((px - mean) / std).to(dtype))[:n].float().cpu().numpy()
        return prep, forward
    raise ValueError(f"unknown encoder {name}")


def encode_view(prep, forward, view, video, split, rows, faces, batch, workers):
    """
    Description: Encodes one split for one view, averaging each clip's vectors over its frames (for the face
    view, over the frames with a face). Up to `workers` upcoming batches are decoded, cropped and preprocessed
    on loader threads while the GPU runs the current one.
    Inputs:
    - prep, forward: the functions from load_encoder.
    - view: "frame" or "face".
    - video: True when prep and forward take whole clips as videos.
    - split: "train", "dev" or "test".
    - rows: line dicts whose frames exist.
    - faces: the dict from find_faces.
    - batch: clips per GPU batch.
    - workers: loader threads.
    Outputs: (a float32 array with one row per clip, which for the face view ends with a has_face column of
    1.0 or 0.0 and holds zeros for clips with no face, a list of milliseconds per clip).
    Constraints: a clip's milliseconds are its batch's preprocessing time (measured on its loader thread, which
    shares the CPU with the others) plus its forward time, divided by the clips in the batch, so they are
    amortised batch times, not single-clip latency. Image decoding and cropping are not timed.
    """
    from PIL import Image

    def load(chunk):
        items, owners = [], []
        for i, r in chunk:
            frames = [Image.open(p).convert("RGB") for p in frame_paths(split, r["clip"])]
            if video:
                items.append(frames)
                owners.append(i)
            elif view == "frame":
                items.extend(frames)
                owners.extend([i] * len(frames))
            else:
                for img, box in zip(frames, faces[r["clip"]]):
                    if box is not None:
                        items.append(speaker_crop(img, box))
                        owners.append(i)
        t = time.perf_counter()
        inputs = prep(items) if items else None
        return inputs, owners, time.perf_counter() - t

    indexed = list(enumerate(rows))
    chunks = [indexed[i:i + batch] for i in range(0, len(rows), batch)]
    sums, hits, ms = None, np.zeros(len(rows)), []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(load, c) for c in chunks[:workers])
        for k, chunk in enumerate(chunks):
            inputs, owners, prep_s = pending.popleft().result()
            if k + workers < len(chunks):
                pending.append(pool.submit(load, chunks[k + workers]))
            torch.cuda.synchronize()
            t = time.perf_counter()
            if owners:
                with torch.inference_mode():
                    vecs = forward(inputs)
                if sums is None:
                    sums = np.zeros((len(rows), vecs.shape[1]), dtype=np.float64)
                np.add.at(sums, owners, vecs)
                np.add.at(hits, owners, 1)
            torch.cuda.synchronize()
            ms.extend([1000 * (prep_s + time.perf_counter() - t) / len(chunk)] * len(chunk))
            done = min((k + 1) * batch, len(rows))
            if (k + 1) % max(1, 2000 // batch) == 0 or done == len(rows):
                print(f"  {view} {split}: {done} of {len(rows)} clips, {time.perf_counter() - t0:.0f} s", flush=True)
    if sums is None:
        raise RuntimeError(f"no {view} images to encode in {split}")
    x = (sums / np.maximum(hits, 1)[:, None]).astype(np.float32)
    if not np.isfinite(x).all():
        raise RuntimeError(f"non-finite features in {view} {split}; try --dtype fp32")
    if view == "face":
        x = np.hstack([x, (hits > 0).astype(np.float32)[:, None]])
    return x, ms


def save_npz(encoder, view, split, rows, x):
    """
    Description: Writes FEATS/<encoder>/<view>_<split>.npz with arrays x, clips and labels, the same layout as
    features.save, through a temporary file so a killed run never leaves a partial file behind.
    Inputs:
    - encoder: the encoder's short name.
    - view: the view name.
    - split: "train", "dev" or "test".
    - rows: line dicts in the order of x's rows.
    - x: the feature array.
    Outputs: none
    """
    folder = FEATS / encoder
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{view}_{split}.part.npz"
    np.savez(tmp, x=x, clips=np.array([r["clip"] for r in rows]), labels=np.array([r["label"] for r in rows]))
    os.replace(tmp, folder / f"{view}_{split}.npz")


def update_timing(entries):
    """
    Description: Merges timing entries into RESULTS/vision_timing.json, keeping other scripts' entries.
    Inputs:
    - entries: a dict from "<encoder>/<view>" to its timing dict.
    Outputs: none
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "vision_timing.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(entries)
    tmp = RESULTS / f"vision_timing.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def timing(ms, device, **extra):
    """
    Description: Summarises per-clip milliseconds.
    Inputs:
    - ms: a list of milliseconds per clip.
    - device: the device name.
    - extra: further fields to keep with the entry.
    Outputs: a dict with ms_median, ms_p95, clips and device.
    """
    return {"ms_median": round(float(np.median(ms)), 2), "ms_p95": round(float(np.percentile(ms, 95)), 2),
            "clips": len(ms), "device": device, **extra}


def main():
    """
    Description: Finds faces in every chosen split, writes faces_stats.json, then extracts each chosen encoder's
    views, skipping any (encoder, view, split) whose file exists. An encoder that fails to load, or a (view,
    split) that fails to encode, is logged to vision_features_log.jsonl and skipped, and the rest still run.
    Inputs: none; reads --encoders (comma-separated short names, "all", or "faces" for face finding only),
    --splits, --batch (clips per GPU batch), --dtype and --workers (YuNet and loader threads) from the command line.
    Outputs: none
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="all")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    encoders = (list(ENCODERS) if args.encoders == "all" else
                [] if args.encoders == "faces" else [e for e in args.encoders.split(",") if e])
    splits = [s for s in args.splits.split(",") if s]
    unknown = [e for e in encoders if e not in ENCODERS] + [s for s in splits if s not in ("train", "dev", "test")]
    if unknown:
        ap.error(f"unknown encoders or splits: {', '.join(unknown)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS / "faces_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["speaker_rule"] = SPEAKER_RULE
    stats["detector"] = {"repo": YUNET_REPO, "file": YUNET_FILE, "score_threshold": FACE_SCORE}
    data, face_ms, face_s = {}, [], 0.0
    for split in splits:
        rows, no_frames = clips_with_frames(split)
        faces, ms, took = find_faces(split, rows, args.workers)
        face_ms.extend(ms)
        face_s += took
        data[split] = (rows, faces)
        stats[split] = face_stats(split, rows, faces, no_frames)
        print(f"{split}: {stats[split]}", flush=True)
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if face_ms:
        update_timing({"yunet/face-finding": timing(face_ms, "cpu", frames_per_clip=N_FRAMES, threads=args.workers,
                                                    clips_per_s=round(len(face_ms) / face_s, 2),
                                                    note="per-clip time inside one of the parallel threads")})
    if not encoders:
        return
    dtype = pick_dtype(args.dtype)
    device = f"{torch.cuda.get_device_name(0)} {str(dtype).replace('torch.', '')}"
    for name in encoders:
        todo = [(v, s) for v in VIEWS[name] for s in splits if not (FEATS / name / f"{v}_{s}.npz").exists()]
        if not todo:
            print(f"{name}: all files exist, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        rec = {"encoder": name, "repo": ENCODERS[name], "dtype": str(dtype).replace("torch.", "")}
        prep = forward = None
        try:
            prep, forward = load_encoder(name, dtype)
            rec["load_s"] = round(time.perf_counter() - t0, 1)
        except Exception:
            rec["error"] = traceback.format_exc()[-1500:]
        done = {}
        for view, split in todo if forward else []:
            rows, faces = data[split]
            try:
                x, ms = encode_view(prep, forward, view, name == "vjepa2-vitl", split, rows, faces, args.batch,
                                    args.workers)
                save_npz(name, view, split, rows, x)
            except Exception:
                rec.setdefault("file_errors", []).append({"view": view, "split": split,
                                                          "error": traceback.format_exc()[-1500:]})
                print(f"{name} {view} {split}: FAILED\n{rec['file_errors'][-1]['error']}", flush=True)
                torch.cuda.empty_cache()
                continue
            done.setdefault(view, ([], []))
            done[view][0].extend(ms)
            done[view][1].append(split)
            rec.setdefault("files", []).append({"view": view, "split": split, "shape": list(x.shape),
                                                "clips_without_frames": stats[split]["clips_without_frames"]})
            print(f"{name} {view} {split}: {x.shape}", flush=True)
        note = "amortised batch time per clip (preprocessing plus forward), image decoding and cropping excluded"
        update_timing({f"{name}/{v}": timing(ms, device, batch_clips=args.batch, note=note, splits=s)
                       for v, (ms, s) in done.items()})
        rec["seconds"] = round(time.perf_counter() - t0)
        rec["gpu_mib_allocated_peak"] = round(torch.cuda.max_memory_allocated() / 2**20)
        print(rec, flush=True)
        with (RESULTS / "vision_features_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        prep = forward = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
