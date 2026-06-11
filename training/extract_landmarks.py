"""Extract MediaPipe landmarks from a directory of labelled images.

Produces CSV rows of ``label, source, frame_id, f0 ... f62`` (one-handed). Source
images are processed once; the resulting CSV is reused across training runs.

    python -m training.extract_landmarks \
        --src datasets/asl_dataset2/asl_alphabet_train/asl_alphabet_train \
        --dst languages/asl/landmarks.csv \
        --hands 1 --source-tag kaggle_asl
"""
import argparse
import csv
import sys
from pathlib import Path

# Allow `python training/extract_landmarks.py ...` (script form) to resolve packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.capture.hands import HandTracker
from src.features.normalise import normalise_one_hand

# Kaggle ASL set includes non-letter folders; skip them for Module 1.
_NON_LETTER = {"DEL", "NOTHING", "SPACE"}


def extract_from_image_dir(
    src_dir: Path, hands: int, source_tag: str, out_csv: Path,
    max_per_class: int | None = None,
) -> tuple[int, int]:
    tracker = HandTracker(static_image_mode=True)
    written = skipped = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for letter_dir in sorted(src_dir.iterdir()):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name.upper()
            if letter in _NON_LETTER or len(letter) != 1:
                continue
            kept = 0
            for img_path in sorted(letter_dir.glob("*.[jp][pn]g")):
                if max_per_class is not None and kept >= max_per_class:
                    break
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                detections = tracker.process(rgb)
                if hands == 1:
                    if not detections:
                        skipped += 1
                        continue
                    feat = normalise_one_hand(detections[0][1])
                else:
                    raise ValueError("Only one-handed extraction is supported.")
                writer.writerow([letter, source_tag, img_path.stem, *feat.tolist()])
                written += 1
                kept += 1
            print(f"  {letter}: kept={kept} (running total written={written})")
    print(f"Wrote {written}, skipped {skipped} (no hand detected)")
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--hands", type=int, choices=[1], default=1)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="cap kept samples per letter (subsampling)")
    args = parser.parse_args()
    extract_from_image_dir(args.src, args.hands, args.source_tag, args.dst,
                           max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
