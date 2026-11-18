"""Predict the emotion of one audio file with the trained model.

Usage: python predict.py path/to/audio.wav [--model_dir outputs]
"""
import argparse
import os

import keras
import librosa
import numpy as np

from emotion_recognition import SR, extract_features

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model_dir", default="outputs")
args = ap.parse_args()

model = keras.models.load_model(os.path.join(args.model_dir, "emotion_model.keras"))
stats = np.load(os.path.join(args.model_dir, "norm_stats.npz"), allow_pickle=True)

y, _ = librosa.load(args.audio, sr=SR)
y, _ = librosa.effects.trim(y, top_db=25)
x = (extract_features(y)[None] - stats["mean"]) / stats["std"]
probs = model.predict(x, verbose=0)[0]

for i in probs.argsort()[::-1]:
    print(f"{stats['classes'][i]:>10}: {probs[i]:.1%}")
