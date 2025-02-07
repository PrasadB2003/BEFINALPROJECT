import time
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation, BatchNormalization, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from config import SAVE_DIR_PATH, MODEL_DIR_PATH

class TrainModel:

    @staticmethod
    def train_neural_network(X, y) -> None:
        """Enhanced neural network training with improved architecture and training regimen."""
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                           random_state=42, stratify=y)

        # Add channel dimension
        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        # Enhanced architecture
        model = Sequential([
            Conv1D(64, 5, padding='same', input_shape=(40, 1)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Conv1D(128, 3, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Flatten(),
            
            Dense(256),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(8, activation='softmax')
        ])

        # Custom Adam optimizer
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1),
            ModelCheckpoint(os.path.join(MODEL_DIR_PATH, 'best_model.h5'), 
                           monitor='val_accuracy', save_best_only=True)
        ]

        # Train model with class weights
        cnn_history = model.fit(x_traincnn, y_train,
                               batch_size=32,
                               epochs=100,
                               validation_data=(x_testcnn, y_test),
                               callbacks=callbacks,
                               class_weight=class_weights)

        # Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(cnn_history.history['loss'], label='Train Loss')
        plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Evolution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
        plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Evolution')
        plt.legend()
        
        plt.savefig('training_metrics.png')
        plt.close()

        # Load best model for evaluation
        model.load_weights(os.path.join(MODEL_DIR_PATH, 'best_model.h5'))
        
        # Evaluation
        predictions = np.argmax(model.predict(x_testcnn), axis=-1)
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

        # Save final model
        model.save(os.path.join(MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model1.h5'))

if __name__ == '__main__':
    print('Training started')
    X = joblib.load(os.path.join(SAVE_DIR_PATH, 'X.joblib'))
    y = joblib.load(os.path.join(SAVE_DIR_PATH, 'y.joblib'))
    TrainModel.train_neural_network(X=X, y=y)