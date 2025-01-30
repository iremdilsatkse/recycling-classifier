import os
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.model_path = 'modelim.h5'
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

    def create_model(self):
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        
        # Unfreeze the top layers of the base model for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, data_path='dataset'):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            subset='validation'
        )

        self.create_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=30,
            callbacks=[early_stopping, reduce_lr]
        )

        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        return history

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("No saved model found. Please train the model first.")

    def fine_tune(self, new_data, new_labels, epochs=1):
        if self.model is None:
            print("Model not loaded. Please load or create the model first.")
            return

        # Convert labels to one-hot encoded format
        one_hot_labels = tf.keras.utils.to_categorical(new_labels, num_classes=len(self.class_names))

        # Fine-tune the model
        history = self.model.fit(
            new_data,
            one_hot_labels,
            epochs=epochs,
            batch_size=4,  # Small batch size for fine-tuning
            verbose=0
        )

        # Save the updated model
        self.model.save(self.model_path)
        print(f"Model fine-tuned and saved to {self.model_path}")

        return history

    def predict(self, image):
        if self.model is None:
            print("Model not loaded. Please load or create the model first.")
            return None

        # Ensure the image is in the correct shape (224, 224, 3) and normalized
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = tf.expand_dims(image, 0)  # Add batch dimension

        predictions = self.model.predict(image)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = float(predictions[0][predicted_class])

        return self.class_names[predicted_class], confidence

