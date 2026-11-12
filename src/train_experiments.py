"""
Phase 3 and 4: Train with different Dropout configurations
Includes Callbacks: ModelCheckpoint and TensorBoard
"""
import numpy as np
import os
from datetime import datetime

# TensorFlow 2.x compatible import
try:
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
except ImportError:
    from keras.callbacks import ModelCheckpoint, TensorBoard

from model_architecture import create_model

print("=" * 50)
print("PHASE 3-4: TRAINING WITH EXPERIMENTS")
print("=" * 50)

# Load data
print("\nLoading training and validation data...")
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_val = np.load('../data/X_val.npy')
y_val = np.load('../data/y_val.npy')

print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")

# Experiment configuration
dropout_values = [0.2, 0.5, 0.8]
epochs = 50
batch_size = 32

print(f"\nExperiments to run: {len(dropout_values)}")
print(f"Dropout rates: {dropout_values}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")

# Create folders for models and logs
os.makedirs('../models', exist_ok=True)
os.makedirs('../logs', exist_ok=True)

# Train one model per dropout value
for dropout in dropout_values:
    print("\n" + "=" * 50)
    print(f"EXPERIMENT: Dropout = {dropout}")
    print("=" * 50)

    # Create model
    model = create_model(dropout_rate=dropout)

    # Configure callbacks
    # 1. ModelCheckpoint: Save best model
    checkpoint_path = f'../models/best_model_dropout_{dropout}.keras'
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # 2. TensorBoard: Logs for visualization
    log_dir = f'../logs/dropout_{dropout}'
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False
    )

    print(f"\nCallbacks configured:")
    print(f"  - ModelCheckpoint: {checkpoint_path}")
    print(f"  - TensorBoard: {log_dir}")

    # Train model
    print(f"\nStarting training ({epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, tensorboard],
        verbose=1
    )

    # Show final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal results:")
    print(f"  Train Accuracy: {final_train_acc:.4f}")
    print(f"  Val Accuracy: {final_val_acc:.4f}")
    print(f"  Difference: {abs(final_train_acc - final_val_acc):.4f}")

print("\n" + "=" * 50)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 50)
print("\nTo visualize in TensorBoard, run:")
print("  tensorboard --logdir=../logs")
print("\nThen open: http://localhost:6006")
