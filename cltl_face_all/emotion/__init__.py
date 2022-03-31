import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

HERE = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


class EmotionDetection:

    def __init__(self, device='cpu'):
        # parameters for loading data and images
        emotion_model_path = 'cltl_face_all/emotion/models/_mini_XCEPTION.102-0.66.hdf5'

        # loading model
        emotion_classifier = load_model(emotion_model_path)

        self.emotion_classifier = emotion_classifier

    def predict(self, faces):
        assert faces.dtype == np.dtype('uint8'), "dtype should be unit8!"

        assert (faces.shape[1], faces.shape[2],
                faces.shape[3]) == (112, 112, 3), "Faces should be cropped "
        "and aligned with the shape of 112 by 112 and RGB!"

        # Remove batch number from faces if 1
        # faces = np.squeeze(faces)
        labeled_preds = []

        for face in faces:
            # Transform Region of Interest to use for the prediction
            roi = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            labeled_preds.append(zip(EMOTIONS, preds))

        return labeled_preds

    def predict_highest(self, faces):
        preds = self.predict(faces=faces)
        res = []
        for pred in preds:
            res.append(max(pred, key=lambda x: x[1])[0])
        return res

