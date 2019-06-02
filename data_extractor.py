import cv2
import os
import uuid
import numpy as np
from PIL import Image

from face_recognizer import crop_image

'''
this python code intends to extract person's faces from video
so that it is possible to treat them as training data
'''


def main(path, label):
    video = cv2.VideoCapture(path)
    out_dir = os.path.join('data/train', '%02d' % (label))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # face recognition
        faces = crop_image(frame)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            image = cv2.resize(frame[y: y + h, x: x + w], (200, 200),
                               interpolation=cv2.INTER_LINEAR)
            image_pil = Image.fromarray(np.uint8(image)).convert('L')
            image_pil.save(os.path.join(out_dir, str(uuid.uuid4()) + '.png'))
            # TODO: flip or limit face crop region

    video.release()


if __name__ == '__main__':
    video_path = 'data/movies/video-1559484434.mp4'
    main(video_path, 16)
