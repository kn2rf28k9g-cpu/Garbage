import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime
import os

print("=" * 60)
print("🚀 Garbage Classification Model - Verbessertes Training")
print("=" * 60)

# Konfiguration
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.0001

# ============================================================
# WICHTIG: Passe diesen Pfad an!
# ============================================================
# Basierend auf deiner Ordnerstruktur:
TRAIN_DIR = 'dataset/train'  # Angepasst an deine Struktur

# Falls die Klassen direkt in dataset/ liegen:
# TRAIN_DIR = 'dataset'

# Absoluter Pfad (Alternative):
# TRAIN_DIR = '/Users/jonasgasparini/PycharmProjects/garbage-classification-model/dataset/train'

# Prüfe ob Verzeichnis existiert
if not os.path.exists(TRAIN_DIR):
    print(f"\n❌ FEHLER: Verzeichnis '{TRAIN_DIR}' nicht gefunden!")
    print("\n🔍 Suche nach möglichen Verzeichnissen...")

    # Suche nach dataset-Verzeichnissen
    current_dir = os.getcwd()
    print(f"   Aktuelles Verzeichnis: {current_dir}")

    possible_paths = [
        'dataset-resized',
        'dataset-resized/train',
        'dataset',
        'data/train',
        '../dataset-resized'
    ]

    print("\n   Prüfe folgende Pfade:")
    for path in possible_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {path}")

    print("\n💡 Lösung: Passe TRAIN_DIR im Code an (Zeile 19-27)")
    exit(1)

print(f"✅ Verzeichnis gefunden: {TRAIN_DIR}")

NUM_CLASSES = 6

# GPU-Check
print(f"\n🎮 GPU gefunden: {len(tf.config.list_physical_devices('GPU'))} Gerät(e)")

# ============================================================
# 1. Data Augmentation und Generatoren
# ============================================================
print("\n📊 Erstelle Daten-Generatoren mit Validation Split...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% für Validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Nur Rescaling für Validation (keine Augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation Generator
validation_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✅ Trainingssamples: {train_generator.samples}")
print(f"✅ Validierungssamples: {validation_generator.samples}")
print(f"✅ Klassen: {train_generator.class_indices}")

# ============================================================
# 2. Class Weights berechnen (für unbalancierte Daten)
# ============================================================
print("\n⚖️ Berechne Class Weights für unbalancierte Klassen...")

class_counts = np.zeros(NUM_CLASSES)
for class_name, class_idx in train_generator.class_indices.items():
    class_dir = os.path.join(TRAIN_DIR, class_name)
    class_counts[class_idx] = len(os.listdir(class_dir))

total_samples = class_counts.sum()
class_weights = {i: total_samples / (NUM_CLASSES * count)
                 for i, count in enumerate(class_counts)}

print("Class Weights:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"   {class_name}: {class_weights[class_idx]:.2f}")

# ============================================================
# 3. Modell erstellen
# ============================================================
print("\n📦 Erstelle Modell-Architektur...")

# Base Model (InceptionV3)
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Friere Base Model zunächst ein
base_model.trainable = False

# Custom Top Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ============================================================
# 4. Callbacks
# ============================================================
print("\n⚙️ Konfiguriere Callbacks...")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/fit/{timestamp}"

callbacks = [
    # Speichert das beste Modell
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Stoppt Training bei Stagnation
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduziert Learning Rate bei Plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    # TensorBoard für Visualisierung
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
]

# Erstelle models Ordner falls nicht vorhanden
os.makedirs('models', exist_ok=True)

# ============================================================
# 5. Kompiliere und trainiere Modell
# ============================================================
print("\n⚙️ Kompiliere Modell...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

# Model Summary
print("\n📋 Modell-Zusammenfassung:")
model.summary()

print(f"\n🏋️ Phase 1: Training mit gefrorenem Base Model ({INITIAL_EPOCHS} Epochen)...")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# 6. Fine-Tuning (optional)
# ============================================================
print("\n\n🔥 Phase 2: Fine-Tuning - Entfriere obere Layers...")

# Entfriere die oberen Layers des Base Models
base_model.trainable = True

# Friere nur die unteren Layers ein
for layer in base_model.layers[:249]:  # InceptionV3 hat 311 Layers
    layer.trainable = False

# Neu kompilieren mit niedriger Learning Rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"🏋️ Starte Fine-Tuning für {FINE_TUNE_EPOCHS} Epochen...")
print("=" * 60)

history_fine = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# 7. Finale Evaluierung
# ============================================================
print("\n\n📊 Finale Evaluierung...")
print("=" * 60)

# Lade bestes Modell
best_model = tf.keras.models.load_model('models/best_model.keras')

# Evaluiere auf Validation Set
results = best_model.evaluate(validation_generator, verbose=1)

print("\n✅ Finale Metriken:")
print(f"   Loss: {results[0]:.4f}")
print(f"   Accuracy: {results[1]:.4f}")
print(f"   Top-2 Accuracy: {results[2]:.4f}")

# Speichere finales Modell
print("\n💾 Speichere finales Modell...")
model.save('models/final_model.keras')
print("   ✅ Gespeichert als: models/final_model.keras")

print("\n🎉 Training abgeschlossen!")
print("=" * 60)
print(f"\n📊 TensorBoard starten mit: tensorboard --logdir={log_dir}")
print("=" * 60)