import argparse
import cv2
import imageio
import numpy as np
import os
import random
from PIL import Image
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup as bs
from tifffile import imread
from tqdm import tqdm
from typing import List, Tuple

parser = argparse.ArgumentParser(
    description='Extract images from TIF image.')
parser.add_argument('image', help='Path to .tif image.')
parser.add_argument('-o', '--out-dir', default='out',
                    help='output directory. Default: out')
parser.add_argument('-p', '--patch-size', default=512, type=int,
                    help='Patch size. Default: 512')
parser.add_argument('-n', '--number-of-images', default=100, type=int,
                    help="Number of output images, Default: 100")
parser.add_argument('-x', '--xml_path', default='annotations.xml',
                    help='Path to xml file with annotated region. Default: annotations.xml')
parser.add_argument('-f', '--full_img', help='Path where save image with drawn patches', default='full_img.png')


def split_image(image, patch_size):
    images = dict()
    for x in range(image.shape[0] // patch_size):
        for y in range(image.shape[1] // patch_size):
            t = image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
            images[str(x) + '_' + str(y)] = t
    return images


def random_patches(images, number_of_images):
    return random.sample(images, number_of_images)


# def convert_to_hsv(patches):
#     return [(name, cv2.cvtColor(im, cv2.COLOR_BGR2HSV)) for name, im in patches.items()]

#
# def sort_patches(patches, number_of_images):
#     return sorted(patches, key=lambda x: -x[1][:, :, 0].std())[:number_of_images]


def save_images(images: List[Tuple[str, np.array]], dir_path):
    for name, image in tqdm(images):
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = Image.fromarray(image)
        image.save(os.path.join(dir_path, name + '.png'))


def read_xml(xml_path):
    with open(xml_path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
    return bs_content


def str_to_list(points):
    points = points.split(';')
    points = [b.split(',') for b in points]
    return [(int(float(b[0])), int(float(b[1]))) for b in points]


def points_to_mask(points, width, height):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    mask = np.array(img)
    return mask


def get_mask(image_xml, base_image, scale_factor=8):
    ''' do zastanowienia: czy zawsze będzie jeden rak na obrazku? '''
    # points = xml.find_one('polygon')
    points = image_xml.find_all('polygon')[0]
    points = str_to_list(points['points'])
    # print(xml)
    width = int(image_xml['width'])
    height = int(image_xml['height'])
    mask = points_to_mask(points, width, height)

    dim = (mask.shape[1] * scale_factor, mask.shape[0] * scale_factor)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

    mask_image = np.dstack((base_image, mask))

    return mask_image

def get_patches_with_mask(patches):
    return [(name, image[:, :, :3]) for name, image in patches.items() if image[:, :, 3].sum() > 0]

def draw_patches(patches, base_image):
    for idx, patch in enumerate(patches):
        idx += 1
        name, image = patch
        name = '_'.join([i for i in name.split('_')][::-1])

        patches[idx] = (str(idx) + '_' + name, image)
        p = [int(i) * 512 for i in name.split('_')]
        cv2.rectangle(base_image, (p[0] - 10, p[1] - 10), (p[0] + 512 + 10, p[1] + 512 + 10), (0, 0, 0), 15)
        base_image = cv2.putText(base_image, str(idx), (p[0] + 512 + 30, p[1] + 512 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                 25, (255, 0, 0), 25, cv2.LINE_AA)

    return patches, base_image


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.xml_path:
        og_path = args.image.split('/')[-1]
        bs_content = read_xml(args.xml_path)
        base_image = imread(args.image)
        # TODO: zrobić tak, żeby można było wybrać, który obrazek z xmla wybrać
        for image_xml in bs_content.find_all('image'):
            name = image_xml['name'].split('/')[-1].replace('.png', '.tif')

            if name == og_path:
                mask_image = get_mask(image_xml, base_image)
                patches = split_image(mask_image, 512)
                patches = get_patches_with_mask(patches)
                patches = random_patches(patches, args.number_of_images)
                patches.sort(key=lambda x: x[0])
                print([k[0] for k in patches])
                patches, image_with_patches = draw_patches(patches, base_image)

                save_images(patches, args.out_dir)
                imageio.imwrite(args.full_img, image_with_patches)

                # FOR TESTING
                if False:
                    he_image = imread('/content/982-22-HE.tif')
                    he_image = cv2.rotate(he_image, cv2.ROTATE_180)
                    for idx, batch in enumerate(patches):
                        name, image = batch

                        p = [int(i) * 512 for i in name.split('_')[1:]]
                        cv2.rectangle(he_image, (p[0] - 512, p[1] - 512), (p[0] + 512 + 512, p[1] + 512 + 512), (0, 0, 0),
                                      15)
                    imageio.imwrite('/content/gdrive/MyDrive/raw/982-22-HE_copy2.jpg', he_image)

                break

        # get_mask()
        # points = str_to_list(bs_content.find_all('points')[0].text)
        # mask = points_to_mask(points, 10000, 10000)
    else:
        print("No xml file provided")
        # input_image = imread(args.image)
        # splitted_images = split_image(input_image, args.patch_size)
        # converted_images = convert_to_hsv(splitted_images)
        # sorted_images = sort_patches(converted_images, args.number_of_images)
        # save_images(sorted_images, args.out_dir)
        # print(f"Images saved in {args.out_dir}.")
