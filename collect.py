"""
collect.py - capture training / test images through the center box.

Examples:
    python collect.py --category cup --split train
    python collect.py --category cup --split test     (use a DIFFERENT cup!)

Keys:
    SPACE  save one crop
    A      toggle auto-capture (about 3 images per second)
    Q      quit
"""
import argparse
import time

import cv2

from common import CATEGORIES, DATASET_DIR, box_coords, crop_box, open_camera, read_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()

    out_dir = DATASET_DIR / args.split / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.jpg")))

    cap = open_camera()
    auto = False
    last_auto = 0.0
    print(f"Saving to {out_dir}  (already has {count} images)")
    print("SPACE = save | A = auto-capture | Q = quit")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Lost camera feed.")
            break

        save_now = False
        if auto and time.time() - last_auto > 0.33:
            save_now = True
            last_auto = time.time()

        crop = crop_box(frame)  # crop BEFORE drawing, so the box line is not in the image
        if save_now:
            cv2.imwrite(str(out_dir / f"{args.category}_{int(time.time() * 1000)}.jpg"), crop)
            count += 1

        x1, y1, x2, y2 = box_coords(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        status = f"{args.split}/{args.category}: {count} images" + ("  [AUTO]" if auto else "")
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            auto = not auto
        elif key == ord(" "):
            cv2.imwrite(str(out_dir / f"{args.category}_{int(time.time() * 1000)}.jpg"), crop)
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
