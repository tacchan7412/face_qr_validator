import os
from PIL import Image

org_dir = 'data/raw'
# 変換対象ファイルの拡張子
org_ext = 'gif'
# 変換後のファイルを格納するディレクトリ
conv_dir = 'data/processed'
# 変換後のファイルの拡張子
conv_ext = 'png'

# 既存のディレクトリをファイルも含めて削除
# if os.path.exists(conv_dir):
    # shutil.rmtree(conv_dir)
# ディレクトリを作成
# os.mkdir(conv_dir)

# 「.」と拡張子を合わせた文字列長
org_ext_len = len(org_ext) + 1

for dirname, dirnames, filenames in os.walk(org_dir):
    for filename in filenames:
        # 変換対象ファイルのパス
        org_path = org_dir + '/' + filename

        # 返還後のファイルパス
        if len(filename) > org_ext_len and \
            filename[-org_ext_len:] == '.' + org_ext:
            filename = filename[0:-org_ext_len]
        conv_path = conv_dir + '/' + filename + '.' + conv_ext

        try:
            # 変換実行
            Image.open(org_path).save(conv_path)
        except IOError:
            print('cannot convert :', org_path)
