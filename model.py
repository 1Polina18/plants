import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
DATA_DIR = 'data'

def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    num_classes = len(class_names)

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                img = Image.open(image_path)
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                img = img.convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_index)
                print(f"Загружено: {image_path}")
            except Exception as e:
                print(f"Ошибка при загрузке изображения {image_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, num_classes, class_names

images, labels, num_classes, class_names = load_data(DATA_DIR)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode='nearest'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
callbacks = [early_stopping, reduce_lr]

history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=(X_test, y_test), epochs=EPOCHS, callbacks=callbacks)

model.save('plant_recognition_model.keras')
print("Модель сохранена как plant_recognition_model.keras")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучающей выборке')
    plt.plot(history.history['val_loss'], label='Потери на валидационной выборке')
    plt.title('График потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    
    plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
    plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)