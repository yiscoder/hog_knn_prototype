"""
train.py - extract HOG features, choose K, train KNN, evaluate, save model.

    python train.py

What it does:
  1. Loads dataset/train and dataset/test.
  2. Tries several K values using cross-validation on the TRAINING data only.
  3. Trains the final KNN with the best K and evaluates it once on the TEST
     data (different physical objects).
  4. Sets the "unknown" distance threshold and saves everything to models/.

Note: cross-validation scores on the training set look optimistic because
images of the same object appear in several folds. The test score is the honest one.
"""
import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from common import CATEGORIES, DATASET_DIR, K_VALUES, MODEL_PATH, RESULTS_DIR, extract_features


def load_split(split):
    X, y = [], []
    for label, cat in enumerate(CATEGORIES):
        folder = DATASET_DIR / split / cat
        files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(label)
    return np.array(X), np.array(y)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    # ---- load and check data
    X_train, y_train = load_split("train")
    if len(y_train) == 0:
        raise SystemExit("No training images found. Run collect.py first.")
    for i, cat in enumerate(CATEGORIES):
        say(f"train/{cat}: {int((y_train == i).sum())} images")
    if len(set(y_train)) < len(CATEGORIES):
        raise SystemExit("Every category needs training images.")
    n_splits = max(2, min(5, int(np.bincount(y_train).min())))

    # ---- choose K with cross-validation
    say("\nCross-validation accuracy on training data (optimistic, see note at top):")
    best_score, k = -1, None
    for candidate in K_VALUES:
        if candidate > len(y_train):
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        score = cross_val_score(KNeighborsClassifier(n_neighbors=candidate),
                                X_train, y_train, cv=cv).mean()
        say(f"  K={candidate}: {score:.3f}")
        if score > best_score:
            best_score, k = score, candidate
    say(f"\nChosen K: {k}")

    # ---- final model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # ---- test on different physical objects
    X_test, y_test = load_split("test")
    if len(y_test) > 0:
        pred = knn.predict(X_test)
        acc = float((pred == y_test).mean())
        say(f"\nTEST accuracy (unseen objects): {acc:.3f}  ({len(y_test)} images)")
        say("\n" + classification_report(
            y_test, pred, labels=list(range(len(CATEGORIES))),
            target_names=CATEGORIES, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=list(range(len(CATEGORIES))))
        say("Confusion matrix (rows = true, columns = predicted):")
        say(str(cm))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES)
        ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test set)")
        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    else:
        say("\nNo test images found. Collect some with: python collect.py --split test ...")

    # ---- "unknown" threshold: 95th percentile of the mean distance to the K
    # nearest training examples, measured on unseen test objects (or on the
    # training set with leave-one-out if there is no test data).
    if len(y_test) > 0:
        dists, _ = knn.kneighbors(X_test)
    else:
        dists, _ = knn.kneighbors(X_train, n_neighbors=min(k + 1, len(y_train)))
        dists = dists[:, 1:]
    threshold = float(np.percentile(dists.mean(axis=1), 95))
    say(f"\nUnknown threshold (mean neighbor distance): {threshold:.3f}")
    say("You can adjust it live in run.py with the + and - keys.")

    joblib.dump({"knn": knn, "k": k, "threshold": threshold, "categories": CATEGORIES},
                MODEL_PATH)
    say(f"\nModel saved to {MODEL_PATH}")

    (RESULTS_DIR / "last_training_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()