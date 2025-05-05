from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from .model import build_model
from .data_loader import create_data_generators 
from .config import Config
import os
import datetime

def train_model():
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    train_gen, val_gen, class_indices = create_data_generators()
    model = build_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6),
        ModelCheckpoint(Config.MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ]
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // Config.BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history