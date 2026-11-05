"""Labelled, reproducible lighting stress test of the production vision profile.

Photometric simulations are not outdoor camera measurements or blind tests.
Every source requires a human-reviewed grid, provenance, grouping and SHA-256.
No camera, hardware, training, implicit labels or model selection is performed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from recognition.onnx_pipeline.chess_recognizer import ChessRecognizer
from localization.live_vision import observe_board

CONDITIONS = (
    "original", "exposure_minus_half_stop", "exposure_plus_half_stop",
    "warm_lamp", "cool_daylight", "soft_shadow", "side_light", "broad_highlight",
)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    cases, seen = [], set()
    for item in manifest["images"]:
        p = Path(item["image"]).expanduser().resolve()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {p}")
        if digest in seen:
            raise ValueError(f"Duplicate source frame: {p}")
        seen.add(digest)
        grid = item["grid"]
        if len(grid) != 10 or any(len(row) != 9 for row in grid):
            raise ValueError("Ground truth must be 10x9")
        if any(ch not in ".KABNRCPkabnrcp" for row in grid for ch in row):
            raise ValueError("Ground truth contains an unknown or unsupported class")
        if not item.get("label_source") or not item.get("position_id"):
            raise ValueError("Human label provenance and position grouping required")
        cases.append({**item, "image": str(p), "grid": [list(row) for row in grid]})
    if not cases:
        raise ValueError("No labelled images")
    return manifest, cases


def illuminate(image, condition):
    if condition == "original":
        return image.copy()
    # Approximate exposure in linear light; parameters are synthetic, not lux.
    srgb = image.astype(np.float32)/255
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    h, w = image.shape[:2]
    if condition == "exposure_minus_half_stop":
        linear *= 2**(-.5)
    elif condition == "exposure_plus_half_stop":
        linear *= 2**.5
    elif condition == "warm_lamp":
        linear *= np.float32([.90, 1, 1.08])
    elif condition == "cool_daylight":
        linear *= np.float32([1.08, 1, .94])
    elif condition == "soft_shadow":
        linear *= np.linspace(.72, 1, w, dtype=np.float32)[None, :, None]
    elif condition == "side_light":
        linear *= np.linspace(.85, 1.15, w, dtype=np.float32)[None, :, None]
    elif condition == "broad_highlight":
        yy, xx = np.ogrid[:h, :w]
        gain = 1+.20*np.exp(-((xx/w-.6)**2+(yy/h-.4)**2)/.12)
        linear *= gain.astype(np.float32)[:, :, None]
    else:
        raise ValueError(condition)
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= .0031308, linear*12.92,
                       1.055*linear**(1/2.4)-.055)
    return np.clip(np.round(encoded*255), 0, 255).astype(np.uint8)


def compare_grid(truth, prediction):
    errors, counts = [], Counter()
    classes = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in range(10):
        for c in range(9):
            actual, predicted = truth[r][c], prediction[r][c]
            counts["cells"] += 1
            counts["correct_cells"] += actual == predicted
            if actual != ".":
                counts["pieces"] += 1
                counts["correct_pieces"] += actual == predicted
                classes[actual]["total"] += 1
                classes[actual]["correct"] += actual == predicted
                counts["misses"] += predicted == "."
                counts["uncertain_pieces"] += predicted == "x"
                if predicted not in (".", "x"):
                    counts["wrong_side"] += actual.isupper() != predicted.isupper()
                    counts["wrong_type"] += actual.lower() != predicted.lower()
            elif predicted not in (".", "x"):
                counts["false_pieces"] += 1
            if predicted != actual:
                errors.append({"position": f"{chr(97+c)}{r}", "truth": actual, "prediction": predicted})
    return dict(counts), dict(classes), errors


def summarize(records):
    counts, reasons, groups = Counter(), Counter(), defaultdict(list)
    accepted = correct_accepted = exact = 0
    for row in records:
        counts.update(row["counts"])
        groups[row["position_id"]].append(not row["errors"])
        exact += not row["errors"]
        if row["rejection"]:
            reasons[row["rejection"]] += 1
        else:
            accepted += 1
            correct_accepted += not row["errors"]
    n = len(records)
    return {"images": n, "positions": len(groups), **counts,
            "cell_accuracy_pct": 100*counts["correct_cells"]/counts["cells"] if n else None,
            "piece_accuracy_pct": 100*counts["correct_pieces"]/counts["pieces"] if counts["pieces"] else None,
            "exact_boards": exact, "board_exact_pct": 100*exact/n if n else None,
            "positions_with_every_repeat_exact": sum(all(v) for v in groups.values()),
            "accepted_images": accepted, "accepted_exact": correct_accepted,
            "accepted_wrong": accepted-correct_accepted,
            "accepted_coverage_pct": 100*accepted/n if n else None,
            "accepted_board_exact_pct": 100*correct_accepted/accepted if accepted else None,
            "rejections": dict(reasons),
            "p50_inference_ms": float(np.median([r["inference_ms"] for r in records])) if n else None}


def evaluate(manifest_path, output_dir, diagnose_clahe=False):
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT):
        raise ValueError("Evaluation output must be outside the project")
    metadata, cases = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec = ChessRecognizer()
    if rec.pose_session is None or rec.classifier_session is None:
        raise RuntimeError("Required ONNX model missing")
    records = []
    preprocessing = ["production"] + (["diagnostic_legacy_clahe"] if diagnose_clahe else [])
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    with (output_dir/"records.jsonl").open("w", encoding="utf-8") as log:
        for case in cases:
            source = cv2.imread(case["image"])
            if source is None:
                raise ValueError(f"Unreadable image: {case['image']}")
            for condition in CONDITIONS:
                image = illuminate(source, condition)
                for processing in preprocessing:
                    rec.reset_buffer()
                    start = time.perf_counter()
                    def enhanced(warped):
                        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
                        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    observation = observe_board(rec, image, warp_preprocess=(
                        enhanced if processing != "production" else None))
                    grid, confidence, reason = (observation["grid_chars"],
                                               observation["conf_grid"], observation["rejection"])
                    counts, classes, errors = compare_grid(case["grid"], grid)
                    row = {"id": case["id"], "position_id": case["position_id"],
                           "image": case["image"], "source_sha256": case["sha256"],
                           "condition": condition, "processing": processing,
                           "grid": grid, "confidence": confidence,
                           "corner_scores": observation["corner_scores"], "corners": observation["corners"],
                           "recovery": observation["recovery"],
                           "rejection": reason, "counts": counts, "classes": classes, "errors": errors,
                           "inference_ms": (time.perf_counter()-start)*1000}
                    records.append(row)
                    log.write(json.dumps(row, ensure_ascii=False)+"\n")
            log.flush()
            print(json.dumps({"completed_source": case["id"], "records": len(records)}), flush=True)
    models = {}
    for name, path in [("pose", rec.pose_model_path), ("layout", rec.classifier_model_path),
                       ("auxiliary", rec.piece_model_path)]:
        p = Path(path)
        models[name] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
    result = {
        "scope": "Human-labelled development photos plus synthetic lighting perturbations; NOT outdoor validation or blind test",
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "manifest_metadata": {k: v for k, v in metadata.items() if k != "images"},
        "models": models, "conditions": CONDITIONS,
        "by_condition": {processing: {condition: summarize(
            [r for r in records if r["processing"] == processing and r["condition"] == condition])
            for condition in CONDITIONS} for processing in preprocessing},
        "overall": {processing: summarize([r for r in records if r["processing"] == processing])
                    for processing in preprocessing},
        "piece_boards_only": {processing: {condition: summarize([r for r in records
            if r["processing"] == processing and r["condition"] == condition and r["counts"].get("pieces", 0)])
            for condition in CONDITIONS} for processing in preprocessing},
    }
    (output_dir/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-legacy-clahe", action="store_true",
                        help="Offline diagnostic only; does not change the live pipeline")
    args = parser.parse_args()
    result = evaluate(args.manifest, args.output_dir, args.diagnose_legacy_clahe)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
