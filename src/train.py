import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_loader import load_data, split_data
from src.model import build_model

BASE_PATH = os.getcwd()
HEIGHT_COLUMN = 'Height (feet)'

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# Load dataset
df = load_data(BASE_PATH)
train_df, val_df = split_data(df)

# Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col=HEIGHT_COLUMN,
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col=HEIGHT_COLUMN,
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE
)

# Build model
model = build_model()

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/pomegranate_height_model.h5")

print("✅ Model trained and saved!")