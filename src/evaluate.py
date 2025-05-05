import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
import seaborn as sns
from .config import Config
import os
import tensorflow as tf

def plot_training_history(history):
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
  
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "training_history.png"))
    plt.close()

def evaluate_model(model, test_gen, class_indices):
    """Evaluate model and generate comprehensive metrics."""
    os.makedirs("reports", exist_ok=True)
    
   
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=Config.CLASS_NAMES))
    
   
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=Config.CLASS_NAMES,
               yticklabels=Config.CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join("reports", "confusion_matrix.png"))
    plt.close()
    
    
    if len(Config.CLASS_NAMES) <= 10:  
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(Config.CLASS_NAMES):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join("reports", "roc_curves.png"))
        plt.close()
    
   
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(Config.CLASS_NAMES):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred[:, i])
        plt.plot(recall, precision, label=f'{class_name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join("reports", "precision_recall.png"))
    plt.close()
    
    
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'class_indices': class_indices
    }
    
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    return test_acc, test_loss