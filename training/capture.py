"""Self-capture tool (DEV-ONLY, not part of the participant lab).

Runs on a developer workstation with a directly attached webcam and a display —
it uses cv2.VideoCapture(0) and cv2.imshow, which need a real camera device and
a window. It is NOT runnable inside a headless pod. Writes landmark coordinates
only (no images), so there are no face-data privacy concerns.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.registry import load_registry


def capture_letter(
    language_code: str,
    letter: str,
    duration_seconds: float = 10.0,
    condition_label: str = "default",
    output_dir: Path = Path("datasets/self_capture"),
):
    langs = load_registry()
    lang = langs[language_code]
    tracker = HandTracker()
    out_path = output_dir / language_code / "landmarks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    end = time.monotonic() + duration_seconds
    rows = []
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feat = build_feature_vector(lang, tracker.process(rgb))
        if feat is None:
            cv2.putText(frame, "NO HAND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
        else:
            rows.append([letter, condition_label, *feat.tolist()])
            cv2.putText(frame, f"REC {letter}: {len(rows)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    with open(out_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)} samples for {letter} ({condition_label})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True)
    p.add_argument("--letter", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--condition", default="default")
    args = p.parse_args()
    capture_letter(args.lang, args.letter, args.duration, args.condition)


if __name__ == "__main__":
    main()
