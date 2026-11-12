"""
Build the deployable artifact bundle for the web demo.

Runs offline (after training). Produces artifacts/ with:
  production_model.keras, scaler.json, defaults.json,
  feature_schema.json, config.json, manifest.json

Run from the repository root:  python src/build_artifacts.py
"""
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

BEST_DROPOUT = 0.2      # model selected officially (best test accuracy)
TEST_ACCURACY = 0.5651

FEATURE_ORDER = [
    "HomeTarget", "HomeCorners", "HomeFouls", "HomeSaves",
    "AwayTarget", "AwayCorners", "AwayFouls", "AwaySaves",
]

LIMITS = {
    "shots_on_target": [0, 30],
    "corners": [0, 30],
    "fouls": [0, 60],
}


def main() -> None:
    out = BASE_DIR / "artifacts"
    out.mkdir(exist_ok=True)

    model_src = BASE_DIR / "models" / f"best_model_dropout_{BEST_DROPOUT}.keras"
    if not model_src.exists():
        raise FileNotFoundError(f"Missing trained model: {model_src}")
    shutil.copyfile(model_src, out / "production_model.keras")

    with open(BASE_DIR / "data" / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    (out / "scaler.json").write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, indent=2),
        encoding="utf-8",
    )

    df = pd.read_csv(BASE_DIR / "dataset" / "Matches.csv")
    defaults = {
        "home": {
            "shots_on_target": round(float(df["HomeTarget"].mean()), 4),
            "corners": round(float(df["HomeCorners"].mean()), 4),
            "fouls": round(float(df["HomeFouls"].mean()), 4),
        },
        "away": {
            "shots_on_target": round(float(df["AwayTarget"].mean()), 4),
            "corners": round(float(df["AwayCorners"].mean()), 4),
            "fouls": round(float(df["AwayFouls"].mean()), 4),
        },
    }
    (out / "defaults.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")

    (out / "feature_schema.json").write_text(
        json.dumps({"features": FEATURE_ORDER}, indent=2), encoding="utf-8"
    )

    config = {
        "limits": LIMITS,
        "max_name_length": 50,
        "idle_timeout_seconds": 300,
        "max_sessions_per_ip": 3,
        "max_messages_per_minute_per_ip": 120,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    manifest = {
        "model_version": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_file": "production_model.keras",
        "dropout": BEST_DROPOUT,
        "test_accuracy": TEST_ACCURACY,
        "input_features": FEATURE_ORDER,
        "classes": ["H", "D", "A"],
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": "Educational demo. Post-match statistics, not a pre-match prediction.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Artifacts written to", out)
    for path in sorted(out.iterdir()):
        print("  -", path.name)


if __name__ == "__main__":
    main()