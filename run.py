"""
run.py - live recognition through the center box.

    python run.py

Keys:
    +  / -   raise / lower the "unknown" threshold (higher = more accepting)
    Q        quit
"""
from collections import Counter, deque

import cv2
import joblib

from common import MODEL_PATH, SMOOTH_FRAMES, box_coords, crop_box, extract_features, open_camera, read_frame


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No model found. Run train.py first.")
    bundle = joblib.load(MODEL_PATH)
    knn = bundle["knn"]
    categories = bundle["categories"]
    threshold = bundle["threshold"]

    cap = open_camera()
    history = deque(maxlen=SMOOTH_FRAMES)

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Lost camera feed.")
            break

        feats = extract_features(crop_box(frame)).reshape(1, -1)
        dists, _ = knn.kneighbors(feats)
        mean_dist = float(dists.mean())

        if mean_dist > threshold:
            raw = "unknown"
        else:
            raw = categories[int(knn.predict(feats)[0])]
        history.append(raw)

        label, votes = Counter(history).most_common(1)[0]
        confident = label != "unknown" and votes >= len(history) / 2
        color = (0, 255, 0) if confident else (0, 0, 255)

        x1, y1, x2, y2 = box_coords(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"distance {mean_dist:.2f} / threshold {threshold:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("AR prototype", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            threshold *= 1.1
        elif key in (ord("-"), ord("_")):
            threshold /= 1.1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()