import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import pathlib

# Direktori dataset utama
data_dir = pathlib.Path("dataset")

# Pengaturan parameter
batch_size = 32
img_height = 224
img_width = 224
validation_split = 0.20
test_split = 0.10

# Fungsi untuk memisahkan dataset menjadi train, val, dan test
def get_datasets(data_dir, img_height, img_width, batch_size, validation_split, test_split):
    total_val_split = validation_split + (1 - validation_split) * test_split

    full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=total_val_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=total_val_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
    test_size = int(val_batches * (test_split / total_val_split))

    test_dataset = val_dataset.take(test_size)
    val_dataset = val_dataset.skip(test_size)

    return full_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_datasets(
    data_dir, img_height, img_width, batch_size, validation_split, test_split
)

# Normalisasi dan augmentasi data
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

def preprocess_data(image, label):
    image = data_augmentation(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

normalized_train_ds = train_dataset.map(preprocess_data)
normalized_val_ds = val_dataset.map(preprocess_data)
normalized_test_ds = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
normalized_train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalized_val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalized_test_ds = normalized_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Menggunakan model pre-trained (Transfer Learning) MobileNetV2
base_model = applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Bekukan lapisan dasar

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(train_dataset.class_names), activation='softmax')
])

model.summary()

# Kompilasi model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Latih model
history = model.fit(
    normalized_train_ds, 
    epochs=15, 
    validation_data=normalized_val_ds,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluasi model
test_loss, test_acc = model.evaluate(normalized_test_ds, verbose=2)
print(f'Test accuracy: {test_acc}')

# Plot akurasi dan loss selama training
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validation')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend(loc='lower right')
plt.title('Akurasi Training dan Validation')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss Training dan Validation')

plt.show()

# Simpan model Keras dalam format .keras
model.save('model.keras')

# Buat input signature
input_shape = (1, 224, 224, 3)
concrete_func = tf.function(lambda x: model(x))
concrete_func = concrete_func.get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))

# Konversi model ke TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Simpan model TFLite ke file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke TFLite dan disimpan sebagai 'model.tflite'")

# Simpan nama-nama kelas ke dalam file labels.txt
labels = train_dataset.class_names

with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

print("Label berhasil disimpan ke 'labels.txt'")