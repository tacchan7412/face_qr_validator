import os
from PIL import Image

org_dir = 'data/raw'
# extension from
org_ext = 'gif'
conv_dir = 'data/processed'
# extension to
conv_ext = 'png'

org_ext_len = len(org_ext) + 1

for dirname, dirnames, filenames in os.walk(org_dir):
    for filename in filenames:
        org_path = org_dir + '/' + filename

        if len(filename) > org_ext_len and \
                filename[-org_ext_len:] == '.' + org_ext:
            filename = filename[0:-org_ext_len]
        conv_path = conv_dir + '/' + filename + '.' + conv_ext

        try:
            Image.open(org_path).save(conv_path)
        except IOError:
            print('cannot convert :', org_path)
