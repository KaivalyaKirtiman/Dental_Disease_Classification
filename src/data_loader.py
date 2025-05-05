import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import Config

def create_data_generators():
    """Create optimized data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(Config.PROCESSED_DATA_DIR, "train"),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(Config.PROCESSED_DATA_DIR, "val"),
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\nClass mapping:")
    for class_name, idx in train_gen.class_indices.items():
        print(f"{class_name} -> {idx}")
    
    return train_gen, val_gen, train_gen.class_indices