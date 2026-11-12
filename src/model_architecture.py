"""
Phase 2: Define Model Architecture
Basic sequential neural network for multiclass classification
"""
import numpy as np

# TensorFlow 2.x compatible import
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
except ImportError:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input

def create_model(dropout_rate=0.5):
    """
    Creates a simple sequential model for football result classification

    Architecture:
    - Input: 8 features
    - Dense(64) + Dropout
    - Dense(32) + Dropout
    - Dense(3, softmax) for 3 classes

    Args:
        dropout_rate: Dropout rate for regularization (0.2, 0.5, 0.8)

    Returns:
        Compiled model
    """
    print(f"\nCreating model with Dropout={dropout_rate}...")

    model = Sequential([
        Input(shape=(8,)),  # 8 input features
        Dense(64, activation='relu', name='hidden_1'),
        Dropout(dropout_rate, name=f'dropout_1_{dropout_rate}'),
        Dense(32, activation='relu', name='hidden_2'),
        Dropout(dropout_rate, name=f'dropout_2_{dropout_rate}'),
        Dense(3, activation='softmax', name='output')  # 3 classes: H, D, A
    ])

    # Compile model
    # Loss: categorical_crossentropy (multiclass classification with one-hot)
    # Optimizer: Adam (adaptive, better than basic SGD)
    # Metrics: accuracy for tracking
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 2: MODEL ARCHITECTURE")
    print("=" * 50)

    # Load data to verify shapes
    print("\nLoading data for verification...")
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # Create test model
    model = create_model(dropout_rate=0.5)

    # Show model summary
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    model.summary()

    # Count total parameters
    total_params = model.count_params()
    print("\n" + "=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Allowed limit: 200,000")

    if total_params <= 200000:
        print("OK: MODEL MEETS THE LIMIT")
    else:
        print("WARNING: MODEL EXCEEDS THE LIMIT")

    print("=" * 50)
    print("\nPHASE 2 COMPLETE")
    print("=" * 50)
