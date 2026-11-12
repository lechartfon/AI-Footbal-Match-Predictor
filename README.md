# Football Match Result Predictor

Neural network for predicting football match outcomes (Home Win, Draw, Away Win) based on historical statistics.

## Project Description

This project implements a simple **sequential neural network** using TensorFlow/Keras for multiclass classification of football match results.

### Features

- Neural network with **&lt; 200,000 parameters**
- **Dropout layers** for regularization
- **3 experiments** with different dropout rates (0.2, 0.5, 0.8)
- **ModelCheckpoint** to save the best model
- **TensorBoard** for monitoring
- Interactive console interface
## Model Features

The model uses **8 features** (4 per team):

**Home Team:**

1. Shots on target (HomeTarget)
2. Corners (HomeCorners)
3. Fouls (HomeFouls)
4. Saves (HomeSaves, equal to the opponent's shots on target)

**Away Team:**

5. Shots on target (AwayTarget)
6. Corners (AwayCorners)
7. Fouls (AwayFouls)
8. Saves (AwaySaves, equal to the opponent's shots on target)

**Target:** 3 classes (Home Win, Draw, Away Win)

> **Note:** Goals (`FTHome`/`FTAway`) are deliberately **not** used. They
> fully determine the match result, so including them would leak the answer
> and produce a misleading 100% accuracy. Only post-match stats that do *not*
> directly encode the result are kept.

## Network Architecture

```
Input(8) -> Dense(64) + Dropout -> Dense(32) + Dropout -> Dense(3, softmax)

Total parameters: ~2,755 (well below the 200,000 limit)
```

- **Hidden activation:** ReLU
- **Output activation:** Softmax (multiclass classification)
- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy

## Project Structure

```
football-match-predictor/
├── dataset/
│   └── Matches.csv              # Dataset with 230,557 matches
├── src/
│   ├── data_preparation.py      # Phase 1: Data preparation
│   ├── model_architecture.py    # Phase 2: Model architecture
│   ├── train_experiments.py     # Phase 3-4: Training experiments
│   ├── evaluate_test.py         # Phase 5: Test evaluation
│   ├── predict_interactive.py   # Interactive console interface
│   ├── build_artifacts.py       # Builds the web artifact bundle
│   ├── inference.py             # Web inference service (model + scaler)
│   ├── terminal_session.py      # WebSocket terminal state machine
│   └── web_app.py               # FastAPI app (WebSocket + REST API)
├── web/                         # Terminal web UI (xterm.js vendored)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── vendor/xterm/            # xterm.js + fit addon (no CDN)
├── artifacts/                   # Generated web bundle (gitignored)
├── tests/                       # Pytest suite
├── deploy/                      # Dockerfile, docker-compose, Caddyfile
├── data/                        # Processed data (train/val/test)
├── models/                      # Saved models (.keras)
├── logs/                        # TensorBoard logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lechartfon/AI-Footbal-Match-Predictor.git
cd AI-Footbal-Match-Predictor
```

### 2. Install dependencies

**From requirements.txt**

```bash
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- TensorFlow 2.15.0
- pandas 2.1.0
- numpy 1.24.3
- scikit-learn 1.3.0

## Usage

### Step 1: Prepare data

```bash
cd src
python data_preparation.py
```

This script:

- Loads the `Matches.csv` dataset
- Selects the 8 features
- Removes NaN values
- Encodes the target (H/D/A)
- Splits into Train (70%), Validation (15%), Test (15%)
- Normalizes with StandardScaler
- Saves data to `../data/`

### Step 2: Verify architecture

```bash
python model_architecture.py
```

Shows the model summary and verifies it meets the parameter limit.

### Step 3: Train experiments

```bash
python train_experiments.py
```

Trains 3 models with different dropout rates:

- Dropout 0.2 (low regularization)
- Dropout 0.5 (medium regularization)
- Dropout 0.8 (high regularization)

Each model trains for **50 epochs** with:

- Batch size: 32
- ModelCheckpoint (saves best model)
- TensorBoard (logs for visualization)

**Estimated time:** 10-30 minutes depending on hardware

### Step 4: Visualize with TensorBoard

```bash
tensorboard --logdir=../logs
```

Open browser at: [http://localhost:6006](http://localhost:6006)

Here you can compare the **3 validation curves** overlaid to choose the best dropout rate.

### Step 5: Evaluate on Test

```bash
python evaluate_test.py
```

Loads the 3 saved models, evaluates on the Test set, and shows:

- Accuracy and Loss for each model
- Best model
- Confusion matrix
- Per-class accuracy

### Step 6: Interactive predictions

```bash
python predict_interactive.py
```

Console interface that allows:

1. Enter team names (e.g., Barcelona vs Real Madrid)
2. Enter statistics (or use dataset averages)
3. Get prediction with percentages for each outcome

**Example:**

```
PREDICTION: Barcelona vs Real Madrid
==========================================================
Barcelona Win        : 45.32% ######################
Draw                 : 28.15% ##############
Real Madrid Win      : 26.53% #############
==========================================================

Prediction: Barcelona WINS
```

## Data Split

After removing rows with missing stats, **113,160 matches** remain:

- **Train:** 70% (79,257 samples) - For training the model
- **Validation:** 15% (16,929 samples) - For hyperparameter tuning
- **Test:** 15% (16,974 samples) - For final evaluation

## Expected Results

The model typically achieves:

- **Test Accuracy:** 55-60%
- **Train Accuracy:** 54-56%

**Note:** Predicting football results is very difficult (many uncaptured variables). 50% accuracy is considered good since random chance gives ~33% (3 classes).

## Web Demo

A public demo is served as a terminal-like web UI (no TensorBoard, no shell,
no training). Flow: educational popup -> terminal -> chat with the predictor
-> server result.

### 1. Build the artifacts (offline, after training)

```bash
python src/build_artifacts.py
```

Produces `artifacts/` with `production_model.keras`, `scaler.json`,
`defaults.json`, `feature_schema.json`, `config.json` and `manifest.json`.

### 2. Run locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

### 3. Test

```bash
python -m pytest tests -q
```

### 4. Deploy

```bash
cd deploy
docker compose up -d --build
```

`docker-compose.yml` publishes only on `127.0.0.1:8000`; put Caddy (or Nginx)
in front with `deploy/Caddyfile` and map `aimatchpredictor.luanef.dev`.

### Web notes

- Single uvicorn worker. Multiple WebSockets run concurrently; the only
  blocking call (`model.predict`) is delegated to the executor via
  `InferenceService.predict_async`, so it never blocks the event loop.
- No training, no subprocess, no shell, no admin endpoints.
- Per-IP rate limiting and idle-session timeouts are configured in
  `artifacts/config.json`.

## Callbacks Implemented

### ModelCheckpoint

```python
ModelCheckpoint(
    filepath='../models/best_model_dropout_X.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

Automatically saves the model only when validation accuracy improves.

### TensorBoard

```python
TensorBoard(
    log_dir='../logs/dropout_X',
    histogram_freq=0
)
```

Logs metrics for real-time visualization.

## Important Notes

1. **Simplicity:** The code is designed to be basic and understandable, following TensorFlow/Keras theory.
2. **Parameter limit:** The model has only ~2,755 parameters, well below the 200,000 limit.
3. **Dropout:** 3 values are tested (0.2, 0.5, 0.8) to find the best balance between underfitting and overfitting.
4. **Normalization:** All features are normalized (mean=0, std=1) to improve training.
5. **One-hot encoding:** The target is encoded as:
  - H (Home Win) -&gt; [1, 0, 0]
  - D (Draw) -&gt; [0, 1, 0]
  - A (Away Win) -&gt; [0, 0, 1]

## Contributing

This is an educational project. Developed as part of the Deep Learning course (UT5).

## License

MIT License

## Author

Luan Charton - AI Football Match Predictor

---

**Enjoy predicting football results with AI!**