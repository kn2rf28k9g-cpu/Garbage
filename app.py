import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
from tensorflow.keras.applications import InceptionV3


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Basis-Modell laden
base_model = InceptionV3(weights='imagenet', include_top=False)

# Eigene Layers hinzufügen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)  # 6 Kategorien

model = Model(inputs=base_model.input, outputs=predictions)

# Modell kompilieren
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_generator, epochs=20)
model.save('garbage_model.h5')