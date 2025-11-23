import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime
import os

print("=" * 60)
print("🔄 Garbage Classification Model - Weitertrainieren")
print("=" * 60)

# Konfiguration
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 10  # Wie viele Epochen ZUSÄTZLICH trainieren?
LEARNING_RATE = 0.00005  # Niedrigere LR für Fine-Tuning
TRAIN_DIR = 'dataset/train'
NUM_CLASSES = 6

# GPU-Check
print(f"\n🎮 GPU gefunden: {len(tf.config.list_physical_devices('GPU'))} Gerät(e)")

# ============================================================
# 1. Lade bestehendes Modell
# ============================================================
print("\n📦 Lade bestehendes Modell...")

MODEL_PATH = 'models/best_model.keras'  # Oder 'models/final_model.keras'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modell geladen von: {MODEL_PATH}")

    # Zeige aktuelle Metriken
    print("\n📊 Modell-Info:")
    print(f"   Total Parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable Parameters: {trainable_params:,}")

except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")
    print("💡 Stelle sicher, dass das Modell existiert!")
    exit(1)

# ============================================================
# 2. Bereite Daten vor (gleich wie beim ersten Training)
# ============================================================
print("\n📊 Bereite Daten vor...")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✅ Trainingssamples: {train_generator.samples}")
print(f"✅ Validierungssamples: {validation_generator.samples}")

# ============================================================
# 3. Class Weights
# ============================================================
print("\n⚖️ Berechne Class Weights...")

class_counts = np.zeros(NUM_CLASSES)
for class_name, class_idx in train_generator.class_indices.items():
    class_dir = os.path.join(TRAIN_DIR, class_name)
    class_counts[class_idx] = len(os.listdir(class_dir))

total_samples = class_counts.sum()
class_weights = {i: total_samples / (NUM_CLASSES * count)
                 for i, count in enumerate(class_counts)}

# ============================================================
# 4. Optional: Entfiere mehr Layers für Fine-Tuning
# ============================================================
print("\n🔓 Optional: Entfiere weitere Layers für besseres Fine-Tuning...")

user_choice = input("Möchtest du mehr Layers trainieren? (j/n): ").lower()

if user_choice == 'j':
    # Finde Base Model (InceptionV3)
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # Das ist das InceptionV3 Model
            # Entfiere alle Layers ab Layer 200 (mehr als vorher)
            for sublayer in layer.layers[200:]:
                sublayer.trainable = True
            print(f"✅ {len([l for l in layer.layers if l.trainable])} Layers des Base Models sind jetzt trainierbar")
            break

    # Zähle trainierbare Parameter
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Neue trainierbare Parameters: {trainable_params:,}")

# ============================================================
# 5. Neu kompilieren mit angepasster Learning Rate
# ============================================================
print("\n⚙️ Kompiliere Modell mit neuer Learning Rate...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"✅ Learning Rate: {LEARNING_RATE}")

# ============================================================
# 6. Callbacks
# ============================================================
print("\n⚙️ Konfiguriere Callbacks...")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/fit/{timestamp}_resume"

callbacks = [
    ModelCheckpoint(
        'models/resumed_best_model.keras',  # Neuer Name um Original nicht zu überschreiben
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
]

# ============================================================
# 7. Evaluiere aktuelles Modell (vor Weitertraining)
# ============================================================
print("\n📊 Evaluiere aktuelles Modell (vor Weitertraining)...")

results_before = model.evaluate(validation_generator, verbose=0)
print(f"   Aktuelle Val-Accuracy: {results_before[1] * 100:.2f}%")
print(f"   Aktuelle Val-Loss: {results_before[0]:.4f}")

# ============================================================
# 8. Weitertrainieren!
# ============================================================
print(f"\n🏋️ Starte Weitertraining für {ADDITIONAL_EPOCHS} zusätzliche Epochen...")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=ADDITIONAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# 9. Evaluiere nach Weitertraining
# ============================================================
print("\n\n📊 Evaluiere nach Weitertraining...")
print("=" * 60)

# Lade bestes Modell
best_model = tf.keras.models.load_model('models/resumed_best_model.keras')
results_after = best_model.evaluate(validation_generator, verbose=1)

print("\n✅ Finale Metriken:")
print(f"   Vorher: {results_before[1] * 100:.2f}%")
print(f"   Nachher: {results_after[1] * 100:.2f}%")
print(f"   Verbesserung: {(results_after[1] - results_before[1]) * 100:+.2f}%")

# Speichere auch als finales Modell
best_model.save('models/final_resumed_model.keras')
print("\n💾 Modelle gespeichert:")
print("   ✅ models/resumed_best_model.keras (bestes während Training)")
print("   ✅ models/final_resumed_model.keras (finales Modell)")

print("\n🎉 Weitertraining abgeschlossen!")
print("=" * 60)
print(f"\n📊 TensorBoard starten mit: tensorboard --logdir={log_dir}")
print("=" * 60)