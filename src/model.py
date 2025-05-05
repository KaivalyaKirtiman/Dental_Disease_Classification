from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from .config import Config

def build_model():
    # Enable mixed precision training if available
    try:
        policy = Policy('mixed_float16')
        set_global_policy(policy)
        print("\nEnabled mixed precision training")
    except:
        print("\nMixed precision not available, using FP32")
    
    # Load base model
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(*Config.IMAGE_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build custom head
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(Config.NUM_CLASSES, activation="softmax", dtype='float32')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model