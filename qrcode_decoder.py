from pyzbar.pyzbar import decode
from PIL import Image


def qr_decode(frame):
    data = decode(frame)
    if len(data) == 0:
        return None
    else:
        return data[0][0].decode('utf-8', 'ignore')


if __name__ == '__main__':
    info = qr_decode(Image.open("data/qrcode/57133810_1218921708272922_5171416846615707648_n.jpg"))
    print(info)
