# HOG + KNN Webcam Prototype

A small experiment: can HOG features + KNN recognize a few object types from a live webcam?
No deep learning. A fixed center box replaces automatic object detection.

## Setup
```
pip install -r requirements.txt
```

## Workflow
1. **Collect training images** (one or two physical examples per category, 30-60 images each,
   vary angle, distance and lighting):
   ```
   python collect.py --category bottle --split train
   python collect.py --category book   --split train
   python collect.py --category cup    --split train
   ```
2. **Collect test images** with a *different* physical object of each category:
   ```
   python collect.py --category bottle --split test
   ...
   ```
   Press SPACE to save one image, or A to toggle auto-capture.
3. **Train and evaluate**:
   ```
   python train.py
   ```
   Prints the HOG-only vs HOG+color comparison, test accuracy and a confusion matrix.
   Output is saved in `results/`.
4. **Run live**:
   ```
   python run.py
   ```
   Hold an object in the box. Use + / - to tune the "unknown" threshold. Q quits.

## Tips
- Use a plain background and steady lighting for the first round.
- Capture images through the same box you use live (the scripts do this for you).
- Show your hand or an unrelated object to check that "unknown" appears.
- Write what you observe in `results/experiment_notes.md`.

## Files
| File | Purpose |
|---|---|
| `common.py` | Shared settings, camera helpers, feature extraction |
| `collect.py` | Capture crops from the webcam |
| `train.py` | Compare settings, train KNN, evaluate, save model |
| `run.py` | Live recognition with unknown rejection and smoothing |
