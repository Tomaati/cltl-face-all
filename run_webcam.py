from cltl_face_all.agegender import AgeGender
from cltl_face_all.arcface import ArcFace
from cltl_face_all.arcface import calc_angle_distance
from cltl_face_all.face_alignment import FaceDetection
from cltl_face_all.emotion import EmotionDetection
from contextlib import contextmanager
import cv2
import numpy as np
from glob import glob
import os


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    # Configure the webcam number if you have more than one.
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yield img_RGB


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def main():
    existing_embs_ = glob(f'your-faces/*/*.npy')
    existing_embs = []
    existing_names = []

    for ee_ in existing_embs_:
        existing_names.append(os.path.basename(ee_).split('.npy')[0])
        existing_embs.append(np.load(ee_).reshape(1, 512))

    existing_embs = np.concatenate(existing_embs, axis=0)

    ag = AgeGender(device='cpu')
    af = ArcFace(device='cpu')
    fd = FaceDetection(device='cpu', face_detector='blazeface')
    ed = EmotionDetection()

    for idx, img in enumerate(yield_images()):
        bboxes = fd.detect_faces(img[np.newaxis, ...])
        landmarks = fd.detect_landmarks(img[np.newaxis, ...], bboxes)
        faces = fd.crop_and_align(img[np.newaxis, ...], bboxes, landmarks)

        # There is only one image per batch. fd returns a list
        bbox = bboxes[0]
        landmark = landmarks[0]
        face = faces[0]

        face_threshold = 0.85

        if len(bbox) > 0:
            print(f"number of faces in this frame: {len(bbox)}")

            # ag and af return a np.ndarray
            age, gender = ag.predict(face)
            embeddings = af.predict(face)
            emotions = ed.predict(bbox, img, face)

            # print(len(bbox), len(landmark), len(face), len(age), len(gender), len(embeddings))

            for bb, lm, a, g, emb in zip(bbox, landmark, age, gender, embeddings):
                x1, y1, x2, y2, prob = bb

                if prob < face_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label = f"{str(round(prob * 100, 1))} % face"
                draw_label(img, (x1, y2), label, font_scale=0.5, thickness=1)

                for lm in landmark:
                    for xy in lm:
                        cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0, 255, 0), -1)

                label = f"{int(a)} years old, {str(round(g * 100, 1))} % female"
                draw_label(img, (x1, y1), label, font_scale=0.5, thickness=1)

                label = f"Feeling {emotions}"
                draw_label(img, (x2, y2), label, font_scale=0.5, thickness=1)

                emb = emb.reshape(1, 512)
                dists = calc_angle_distance(emb, existing_embs)

                candidate, dist = existing_names[np.argmin(dists)], np.min(dists)
                ANGLE_DIST_THRESHOLD = 1.15
                if dist < ANGLE_DIST_THRESHOLD:
                    label = f"{candidate}"
                    draw_label(img, (x2, y2), label, font_scale=0.5, thickness=1)

        cv2.imshow("result", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)

        for idx, fc in enumerate(face):
            cv2.imshow(f"cropped and aligned {idx}", cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
