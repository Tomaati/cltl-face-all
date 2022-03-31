import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

HERE = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
face_threshold = 0.85


def predict_highest(preds):
    return max(preds, key=lambda x: x[1])[0]


class EmotionDetection:

    def __init__(self):
        # parameters for loading data and images
        detection_model_path = 'cltl_face_all/emotion/haarcascade_files/haarcascade_frontalface_default.xml'
        emotion_model_path = 'cltl_face_all/emotion/models/_mini_XCEPTION.102-0.66.hdf5'

        # loading models
        face_detection = cv2.CascadeClassifier(detection_model_path)
        emotion_classifier = load_model(emotion_model_path)

        self.detection_model = face_detection
        self.emotion_classifier = emotion_classifier

    def predict(self, faces, img):
        for array in faces:
            for (fX, fY, fX2, fY2, prob) in array:
                if prob < face_threshold:
                    continue

                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels,
                # and then prepare the ROI for classification via the CNN
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(f"{fX} {fY} {fX2} {fY2}")
                roi = gray[int(fY):int(fY2), int(fX):int(fX2)]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_classifier.predict(roi)[0]
                zipped = zip(EMOTIONS, preds)
                return predict_highest(zipped)

