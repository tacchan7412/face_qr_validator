import cv2
import numpy as np
from PIL import Image

from face_recognizer import FaceRecognizer
from face_recognizer import crop_image
from qrcode_decoder import qr_decode


threshold = 100000


def main(path, correct_label):
    qr_flag = False
    fr = FaceRecognizer()
    video = cv2.VideoCapture(path)

    face_wrong_cnt = 0
    face_undetected_cnt = 0
    multiple_face_cnt = 0
    cnt = 0

    while video.isOpened():
        cnt += 1
        ret, frame = video.read()
        if not ret:
            break

        # QR code detection
        if not qr_flag:
            info = qr_decode(Image.fromarray(np.uint8(frame)))
            if info:
                print('QR code detected')
                print(info)
                qr_flag = True

        # face recognition
        faces = crop_image(frame)
        if len(faces) >= 1:
            x, y, w, h = faces[0]
            image = cv2.resize(frame[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            image = np.mean(image, -1)
            label, _ = fr.recognizer.predict(image)
            if label != correct_label:
                face_wrong_cnt += 1
        elif len(faces) > 1:
            # print('multiple faces are detected!')
            multiple_face_cnt += 1
            # break
        elif len(faces) == 0:
            face_undetected_cnt += 1
        if face_wrong_cnt > threshold:
            print('face recognition failed')
            print(cnt)
            break

    video.release()
    print('total frames', cnt)
    print('wrong', face_wrong_cnt)
    print('undetected', face_undetected_cnt)
    print('multiple', multiple_face_cnt)


if __name__ == '__main__':
    video_path = "data/movies/ayumu.mp4"
    main(video_path, 16)  # 16 is Ayumu
