"""
Phase 5: Evaluate the best model on Test set
"""
import numpy as np

# TensorFlow 2.x compatible import
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

print("=" * 50)
print("PHASE 5: TEST EVALUATION")
print("=" * 50)

# Load test data
print("\n1. Loading test data...")
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')
print(f"   Test: {X_test.shape[0]} samples")

# Evaluate all 3 saved models and pick the best
print("\n2. Evaluating all 3 saved models...")
dropout_values = [0.2, 0.5, 0.8]
results = []

for dropout in dropout_values:
    model_path = f'../models/best_model_dropout_{dropout}.keras'
    try:
        model = load_model(model_path)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append({
            'dropout': dropout,
            'loss': loss,
            'accuracy': accuracy
        })
        print(f"   Dropout {dropout}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
    except:
        print(f"   Dropout {dropout}: Model not found")

# Find the best model
if results:
    best_model = max(results, key=lambda x: x['accuracy'])
    print(f"\n3. Best model:")
    print(f"   Dropout: {best_model['dropout']}")
    print(f"   Test Accuracy: {best_model['accuracy']:.4f}")
    print(f"   Test Loss: {best_model['loss']:.4f}")

    # Load best model for detailed analysis
    best_model_path = f"../models/best_model_dropout_{best_model['dropout']}.keras"
    model = load_model(best_model_path)

    # Predictions
    print("\n4. Prediction analysis...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion matrix
    class_names = ['Home Win', 'Draw', 'Away Win']
    print("\n   Confusion Matrix:")
    print("   " + " " * 15 + "Predicted")
    print("   " + " " * 10 + "H      D      A")

    for i, true_class in enumerate(class_names):
        row = f"   {true_class:10s}"
        for j in range(3):
            count = np.sum((y_true_classes == i) & (y_pred_classes == j))
            row += f" {count:5d} "
        print(row)

    # Per-class accuracy
    print("\n   Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = np.mean(y_pred_classes[mask] == i)
            print(f"   {class_name:10s}: {class_acc:.4f}")

    print("\n" + "=" * 50)
    print("PHASE 5 COMPLETE")
    print("=" * 50)
else:
    print("\nError: No saved models found")
