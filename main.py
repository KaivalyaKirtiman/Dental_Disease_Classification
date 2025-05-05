from src.train import train_model
from src.evaluate import evaluate_model, plot_training_history
from src.data_loader import create_data_generators
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt

def main():
    """Main execution pipeline."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("="*50, "\nStarting model training...\n", "="*50)
    train_gen, val_gen, class_indices = create_data_generators()
    model, history = train_model()
    
    plot_training_history(history)
    
    print("\n" + "="*50, "\nEvaluating model...\n", "="*50)
    test_acc, test_loss = evaluate_model(model, val_gen, class_indices)
    
    print("\nFinal Evaluation:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    tf.keras.backend.clear_session()
    print("\n" + "="*50)
    print("Model training complete!" if test_acc >= 0.75 else "Model needs improvement")
    print("="*50)

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages
    main()