import argparse
import os
from typing import List, Tuple
from tifffile import imread
import cv2
from PIL import Image
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser(
    description='Extract images from TIF image.')
parser.add_argument('image', help='Path to .tif image.')
parser.add_argument('-o', '--out-dir', default='out',
                    help='output directory. Default: out')
parser.add_argument('-p', '--patch-size', default=512, type=int,
                    help='Patch size. Default: 512')
parser.add_argument('-n', '--number-of-images', default=100, type=int,
                    help="Number of output images, Default: 100")


def split_image(image, patch_size):
    images = dict()
    for x in range(image.shape[0] // patch_size):
        for y in range(image.shape[1] // patch_size):
            t = image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
            images[str(x) + '_' + str(y)] = t
    return images


def convert_to_hsv(patches):
    return [(name, cv2.cvtColor(im, cv2.COLOR_BGR2HSV)) for name, im in patches.items()]


def sort_patches(patches, number_of_images):
    return sorted(patches, key=lambda x: -x[1][:, :, 0].std())[:number_of_images]


def save_images(images: List[Tuple[str, np.array]], dir_path):
    for name, image in tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = Image.fromarray(image)
        image.save(os.path.join(dir_path, name + '.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    input_image = imread(args.image)
    splitted_images = split_image(input_image, args.patch_size)
    converted_images = convert_to_hsv(splitted_images)
    sorted_images = sort_patches(converted_images, args.number_of_images)
    save_images(sorted_images, args.out_dir)
    print(f"Images saved in {args.out_dir}.")
