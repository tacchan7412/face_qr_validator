import cv2
import os
import numpy as np
from PIL import Image


cascadePath = "config/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def crop_image(image, scaleFactor=1.11, minNeighbors=3, minSize=(200, 200)):
    faces = faceCascade.detectMultiScale(image, scaleFactor=scaleFactor,
                                         minNeighbors=minNeighbors,
                                         minSize=minSize)
    return faces


class FaceRecognizer():
    def __init__(self):
        train_path = './data/train'

        # EigenFace
        # self.recognizer = cv2.face.EigenFaceRecognizer_create()
        # FisherFace
        self.recognizer = cv2.face.FisherFaceRecognizer_create()
        # LBPH
        # self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.train(train_path)

    def get_images_and_labels(self, path):
        images = []
        labels = []
        files = []
        for f in os.listdir(path):
            if not os.path.isdir(os.path.join(path, f)):
                continue
            else:
                label = int(f)
                for fn in os.listdir(os.path.join(path, f)):
                    image_path = os.path.join(path, f, fn)
                    # to grayscale
                    image_pil = Image.open(image_path).convert('L')
                    image = np.array(image_pil, 'uint8')
                    faces = crop_image(image)
                    for (x, y, w, h) in faces:
                        roi = cv2.resize(image[y: y + h, x: x + w], (200, 200),
                                         interpolation=cv2.INTER_LINEAR)
                        images.append(roi)
                        labels.append(label)
                        files.append(os.path.join(f, fn))

        return images, labels, files

    def train(self, train_path):
        images, labels, files = self.get_images_and_labels(train_path)
        self.recognizer.train(images, np.array(labels))


if __name__ == '__main__':
    test_path = './data/test'
    fr = FaceRecognizer()
    # テスト画像を取得
    test_images, test_labels, test_files = fr.get_images_and_labels(test_path)
    print(len(test_images))

    for i in range(len(test_labels)):
        # テスト画像に対して予測実施
        label, confidence = fr.recognizer.predict(test_images[i])
        # 予測結果をコンソール出力
        print("Test Image: {}, Predicted Label: {}, Confidence: {}"
              .format(test_files[i], label, confidence))
