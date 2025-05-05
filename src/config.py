import os

class Config:
    # Data paths
    BASE_DIR = "data"
    RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed")
    
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32  # Can increase to 64 if you have GPU memory
    NUM_CLASSES = 6
    
    # Training parameters
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    EARLY_STOPPING_PATIENCE = 5
    
    # Model saving (using new .keras format)
    MODEL_SAVE_PATH = "models/resnet50v2_oral_disease.keras"
    
    # Class names (must match your folder names exactly)
    CLASS_NAMES = [
        "Calculus", 
        "Caries", 
        "Gingivitis", 
        "Ulcers", 
        "Tooth-Discoloration", 
        "Hypodontia"
    ]