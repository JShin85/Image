import os
import argparse
from PIL import Image
import random


def merge_images(file_prefix, cols, rows, output_name):

    files = os.listdir()

    image_paths = [file for file in files if file.startswith(f'{file_prefix}')]

    # 각 분할 이미지 크기 가져오기
    example_img = Image.open(image_paths[0])
    width, height = example_img.size
    s, t = width, height

    # 모든 분할 이미지 크기를 일치시키기
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        img = img.resize((s, t), Image.ANTIALIAS)
        img.save(image_path)

    # 병합할 이미지 크기 계산
    merge_width, merge_height = s * cols, t * rows

    # 빈 이미지 생성
    merged_image = Image.new('RGB', (merge_width, merge_height))

    # 분할된 이미지를 병합할 위치 계산
    for j in range(rows):
        for i in range(cols):
            # 이미지 열기
            img = Image.open(image_paths[i + j * cols])

            # 이미지 회전
            if random.random() < 0.5:
                img = img.transpose(Image.ROTATE_90)

            # 이미지 뒤집기
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # 이미지 크기 일치시키기
            img = img.resize((s, t), Image.ANTIALIAS)    

            # 이미지 병합
            left, top = i * s, j * t
            right, bottom = left + s, top + t
            merged_image.paste(img, (left, top, right, bottom))

    # out_dir = f"{cols}x{rows}"
    # if out_dir not in os.listdir():
    #     os.mkdir(out_dir)        

    # 이미지 저장
    # merged_image.save(f"{out_dir}/{output_name}.jpg")
    merged_image.save(f"{output_name}.jpg")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='merge image')
    parser.add_argument('--filenameprefix', type=str, help='input file name prefix')
    parser.add_argument('--cols', type=int, help='column number')
    parser.add_argument('--rows', type=int, help='row number')
    parser.add_argument('--outputname', type=str, help='output file name')
    args = parser.parse_args()

    merge_images(args.filenameprefix, args.cols, args.rows, args.outputname)