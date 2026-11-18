"""
CodeAlpha ML Internship - Task 2: Emotion Recognition from Speech
Dataset : RAVDESS (audio speech, 24 actors, 8 emotions)
Features: MFCC sequences (+ delta) extracted with librosa
Model   : 1D CNN + Bi-LSTM (Keras / TensorFlow)

Usage:
    python emotion_recognition.py --data_dir path/to/RAVDESS --epochs 60

RAVDESS filename format: 03-01-06-01-02-01-12.wav
    field 3 (06) = emotion, last field (12) = actor id
"""

import argparse
import glob
import os

import keras
import librosa
import matplotlib

matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import callbacks, layers, models

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
SR = 22050
DURATION = 3.0  # seconds; shorter clips are padded, longer are cut
N_MFCC = 40
HOP = 512
MAX_FRAMES = int(np.ceil(SR * DURATION / HOP))


def extract_features(y):
    """MFCC + delta MFCC -> array of shape (MAX_FRAMES, 2 * N_MFCC)."""
    target = int(SR * DURATION)
    y = librosa.util.fix_length(y, size=target)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta]).T  # (frames, 80)
    return feat[:MAX_FRAMES]


def augment(y):
    """Return a list of augmented copies: noise, pitch shift, time stretch."""
    noise = y + 0.005 * np.random.randn(len(y))
    pitch = librosa.effects.pitch_shift(y, sr=SR, n_steps=2)
    stretch = librosa.effects.time_stretch(y, rate=0.9)
    return [noise, pitch, stretch]


def load_dataset(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")
    items = []  # (features, label, actor, is_augmented)
    for i, path in enumerate(files, 1):
        parts = os.path.basename(path).replace(".wav", "").split("-")
        if len(parts) != 7 or parts[2] not in EMOTIONS:
            continue
        label, actor = EMOTIONS[parts[2]], int(parts[6])
        y, _ = librosa.load(path, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=25)  # drop leading/trailing silence
        items.append((extract_features(y), label, actor, False))
        for y_aug in augment(y):
            items.append((extract_features(y_aug), label, actor, True))
        if i % 100 == 0:
            print(f"  processed {i}/{len(files)} files")
    X = np.array([it[0] for it in items], dtype="float32")
    labels = np.array([it[1] for it in items])
    actors = np.array([it[2] for it in items])
    aug = np.array([it[3] for it in items])
    return X, labels, actors, aug


def build_model(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.history["loss"], label="train")
    ax[0].plot(h.history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(h.history["accuracy"], label="train")
    ax[1].plot(h.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", required=True, help="Folder containing RAVDESS Actor_XX folders"
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    keras.utils.set_random_seed(42)

    print("Extracting features...")
    X, labels, actors, is_aug = load_dataset(args.data_dir)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Samples: {len(X)}  shape: {X.shape}  classes: {list(le.classes_)}")

    # Speaker-independent split: whole actors go to train / val / test, so the
    # model is evaluated on voices it has never heard (no leakage).
    train_m = actors <= 18
    val_m = (actors >= 19) & (actors <= 21)
    test_m = actors >= 22
    # Augmented copies only used for training; val/test use original clips.
    val_m &= ~is_aug
    test_m &= ~is_aug
    X_tr, y_tr = X[train_m], y[train_m]
    X_va, y_va = X[val_m], y[val_m]
    X_te, y_te = X[test_m], y[test_m]

    # Normalise with training statistics only
    mean, std = (
        X_tr.mean(axis=(0, 1), keepdims=True),
        X_tr.std(axis=(0, 1), keepdims=True) + 1e-8,
    )
    X_tr, X_va, X_te = [(a - mean) / std for a in (X_tr, X_va, X_te)]
    print(f"train {len(X_tr)} | val {len(X_va)} | test {len(X_te)}")

    model = build_model(X_tr.shape[1:], len(le.classes_))
    model.summary()
    cbs = [
        callbacks.EarlyStopping(
            patience=12, restore_best_weights=True, monitor="val_accuracy"
        ),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss"),
    ]
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=2,
    )

    print("\nTest evaluation (unseen speakers):")
    pred = model.predict(X_te).argmax(axis=1)
    print(classification_report(y_te, pred, target_names=le.classes_, digits=3))
    cm = confusion_matrix(y_te, pred)

    plot_history(hist, os.path.join(args.out_dir, "training_curves.png"))
    plot_confusion(cm, le.classes_, os.path.join(args.out_dir, "confusion_matrix.png"))
    model.save(os.path.join(args.out_dir, "emotion_model.keras"))
    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        mean=mean,
        std=std,
        classes=le.classes_,
    )
    print(f"Saved model and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
