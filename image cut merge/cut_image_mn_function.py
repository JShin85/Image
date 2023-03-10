import os
import random
import argparse
from PIL import Image
import string
import time

timestamp = int(time.time())


def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def split_image(imgfile: str, M: int, N: int, output_name_prefix:str) -> None:
    # 이미지 열기
    img = Image.open(imgfile)

    # 이미지 크기 가져오기
    width, height = img.size

    # 분할할 이미지 크기 계산
    s, t = width // M, height // N

    # out_dir = f"{M}x{N}"
    # if out_dir not in os.listdir():
    #     os.mkdir(out_dir)

    for j in range(N):
        for i in range(M):
            # 분할할 이미지 위치 계산
            left, top = i * s, j * t
            right, bottom = left + s, top + t

            # 이미지 분할
            cropped_img = img.crop((left, top, right, bottom))

            random_string = generate_random_string(5)
            # cropped_img.save(f"{out_dir}/{random_string}.jpg")  
            # cropped_img.save(f"{out_dir}/{output_name_prefix}_{timestamp}_{random_string}.jpg")
            cropped_img.save(f"{output_name_prefix}_{M}X{N}_{timestamp}_{random_string}.jpg")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='cut image')
    parser.add_argument('--imgfile', type=str, help='type file name')
    parser.add_argument('--cols', type=int, help='column number')
    parser.add_argument('--rows', type=int, help='row number')
    parser.add_argument('--prefix', type=str, help='type output file name prefix')
    args    = parser.parse_args()

    split_image(args.imgfile, args.cols, args.rows, args.prefix) # split_image('test_img.png', 3, 4, test)
    