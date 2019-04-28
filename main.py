import cv2
import numpy as np
from PIL import Image

from face_recognizer import FaceRecognizer
from face_recognizer import crop_image
from qrcode_decoder import qr_decode


threshold = 100000


def main(path, correct_label):
    qr_flag = False
    correct_label = 16
    fr = FaceRecognizer()
    video = cv2.VideoCapture(path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    face_correct_cnt = 0
    face_wrong_cnt = 0
    face_undetected_cnt = 0
    multiple_face_cnt = 0
    cfd_high = 0
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
        if len(faces) == 1:
            x, y, w, h = faces[0]
            image = cv2.resize(frame[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            image_pil = Image.fromarray(np.uint8(image)).convert('L')
            image = np.array(image_pil, 'uint8')
            label, cfd = fr.recognizer.predict(image)
            if label != correct_label:
                face_wrong_cnt += 1
            else:
                face_correct_cnt += 1
            if cfd > 40:
                cfd_high += 1

            # write rectangle to indicate where the face is
            color = (0, 255, 0) if label == correct_label else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt = 'Ayumu' if label == correct_label else 'not Ayumu'
            cv2.putText(frame, txt + '%.3f' % (cfd), (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

        out.write(frame)

    video.release()
    out.release()
    print('total frames', cnt)
    print('correct', face_correct_cnt)
    print('wrong', face_wrong_cnt)
    print('undetected', face_undetected_cnt)
    print('multiple', multiple_face_cnt)
    # print('confidence high', cfd_high)


if __name__ == '__main__':
    video_path = "data/movies/ayumu.mp4"
    main(video_path, 16)  # 16 is Ayumu
