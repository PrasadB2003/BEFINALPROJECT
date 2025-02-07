"""
This file can be used to try a live prediction.
"""

import os
import keras
import librosa
import numpy as np

from config import EXAMPLES_PATH, MODEL_DIR_PATH


class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = os.path.join(MODEL_DIR_PATH, 'best_model.h5')  # ✅ Fixed Path
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create features.
        """
        data, sampling_rate = librosa.load(self.file, sr=None)  # ✅ Keep original sampling rate
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)

        # Fix input shape for model prediction
        x = np.expand_dims(mfccs, axis=0)  # (1, 40)
        x = np.expand_dims(x, axis=-1)  # (1, 40, 1)

        # ✅ Fix predict_classes() issue
        predictions = np.argmax(self.loaded_model.predict(x), axis=-1)[0]

        print("Prediction is:", self.convert_class_to_emotion(predictions))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Convert predictions (int) into human-readable strings.
        """
        label_conversion = {
            0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
            4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
        }
        return label_conversion.get(pred, "Unknown")


if __name__ == '__main__':
    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, 'YAF_bar_happy.wav'))
    live_prediction.loaded_model.summary()
    live_prediction.make_predictions()

    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, 'OAF_back_fear.wav'))
    live_prediction.make_predictions()


    live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, 'YAF_back_neutral.wav '))
    live_prediction.make_predictions()
