import cv2
import os
import numpy as np
from PIL import Image


cascadePath = "config/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def crop_image(image):
    faces = faceCascade.detectMultiScale(image)
    return faces


class FaceRecognizer():
    def __init__(self):
        # トレーニング画像
        train_path = './data/train'

        # Haar-like特徴分類器

        # 顔認識器の構築 for OpenCV 2
        #   ※ OpenCV3ではFaceRecognizerはcv2.faceのモジュールになります
        # EigenFace
        #recognizer = cv2.createEigenFaceRecognizer()
        # FisherFace
        #recognizer = cv2.createFisherFaceRecognizer()
        # LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.train(train_path)

    # 指定されたpath内の画像を取得
    def get_images_and_labels(self, path):
        # 画像を格納する配列
        images = []
        # ラベルを格納する配列
        labels = []
        # ファイル名を格納する配列
        files = []
        for f in os.listdir(path):
            # 画像のパス
            image_path = os.path.join(path, f)
            # グレースケールで画像を読み込む
            image_pil = Image.open(image_path).convert('L')
            # NumPyの配列に格納
            image = np.array(image_pil, 'uint8')
            # Haar-like特徴分類器で顔を検知
            faces = crop_image(image)
            # 検出した顔画像の処理
            for (x, y, w, h) in faces:
                # 顔を 200x200 サイズにリサイズ
                roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
                # 画像を配列に格納
                images.append(roi)
                # ファイル名からラベルを取得
                labels.append(int(f[7:9]))
                # ファイル名を配列に格納
                files.append(f)

        return images, labels, files

    def train(self, train_path):
        images, labels, files = self.get_images_and_labels(train_path)
        self.recognizer.train(images, np.array(labels))


if __name__ == '__main__':
    test_path = './data/test'
    fr = FaceRecognizer()
    # テスト画像を取得
    test_images, test_labels, test_files = fr.get_images_and_labels(test_path)

    for i in range(len(test_labels)):
        # テスト画像に対して予測実施
        label, confidence = fr.recognizer.predict(test_images[i])
        # 予測結果をコンソール出力
        print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
