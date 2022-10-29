import argparse
import cv2
import imageio
import json
import numpy as np
import os
import random
from PIL import Image
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup as bs
from tifffile import imread
from tqdm import tqdm
from typing import List, Tuple
from openslide import open_slide

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
parser.add_argument('-gd', '--gdrive', help='Path to gdrive folder', default='/content/gdrive/MyDrive/inz/default')


def split_image(image, patch_size):
    images = dict()
    for x in range(image.shape[0] // patch_size):
        for y in range(image.shape[1] // patch_size):
            t = image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
            images[str(x) + '_' + str(y)] = t
    return images


def random_patches(images, number_of_images):
    return random.sample(images, number_of_images)


def save_images(images: List[Tuple[str, np.array]], dir_path):
    for name, image in tqdm(images):
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        # image = Image.fromarray(image)
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


def get_mask(image_xml, scale_factor=16):
    mask = None
    width = int(image_xml['width'])
    height = int(image_xml['height'])
    for point_cloud in image_xml.find_all('polygon'):
        points = str_to_list(point_cloud['points'])
        if mask is None:
            mask = points_to_mask(points, width, height)
        else:
            mask += points_to_mask(points, width, height)

    dim = (mask.shape[1] * scale_factor, mask.shape[0] * scale_factor)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

    return mask


def get_patches_with_mask(patches):
    return [(name, image) for name, image in patches.items() if image.sum() > 0]


def get_true_patches(patches, slide):
    images = []
    for name, image in patches:
        # print(name, patches)
        temp_name = '_'.join([i for i in name.split('_')][::-1])
        p = [int(i) * 512 for i in temp_name.split('_')]
        smaller_region = slide.read_region((p[0], p[1]), 0, (512, 512))
        images.append((name, smaller_region))
    return images


def draw_patches(patches, base_image_path):
    base_image = imread(base_image_path, key=4)
    print(base_image.shape)
    for idx, patch in enumerate(patches):
        name, image = patch
        name = '_'.join([i for i in name.split('_')][::-1])

        patches[idx] = (str(idx + 1) + '_' + name, image)
        p = [int(i) * 128 for i in name.split('_')]
        cv2.rectangle(base_image, (p[0] - 10, p[1] - 10), (p[0] + 128 + 10, p[1] + 128 + 10), (0, 0, 0), 15)
        base_image = cv2.putText(base_image, str(idx + 1), (p[0] + 128 + 30, p[1] + 128 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                 25, (255, 0, 0), 25, cv2.LINE_AA)

    return patches, base_image


def extract_patches_from_image(project_path, output_path, number_of_patches):
    raw_folder = os.path.join(project_path, 'raw')
    files = os.listdir(raw_folder)
    base_image_path = [os.path.join(raw_folder, f) for f in files if 'Ki67' in f][0]
    xml_path = os.path.join(project_path, 'annotations.xml')
    og_path = base_image_path.split('/')[-1]
    JPGS_path = os.path.join(project_path, 'JPGS')
    image_with_patches_path = os.path.join(JPGS_path, og_path.replace('.tif', '.jpg'))
    csv_path = os.path.join(project_path, 'patches.csv')

    os.makedirs(JPGS_path, exist_ok=True)

    bs_content = read_xml(xml_path)

    for image_xml in bs_content.find_all('image'):
        name = image_xml['name'].split('/')[-1].replace('.png', '.tif')

        if name == og_path:

            mask = get_mask(image_xml)
            print(mask.shape)
            patches = split_image(mask, 512)
            patches = get_patches_with_mask(patches)

            patches = random_patches(patches, number_of_patches)
            csv_out = [k[0] for k in patches]
            with open(csv_path, 'w') as f:
                json.dump(csv_out, f, indent=2)

            patches.sort(key=lambda x: x[0])

            slide = open_slide(base_image_path)
            patches = get_true_patches(patches, slide)
            # print(patches)

            patches, image_with_patches = draw_patches(patches, base_image_path)
            save_images(patches, output_path)
            # print(args.full_img, image_with_patches.shape)
            # dim = (image_with_patches.shape[1] // 2, image_with_patches.shape[0] // 2)
            # image_with_patches = cv2.resize(image_with_patches, dim, interpolation=cv2.INTER_AREA)
            # print(image_with_patches_path, image_with_patches.shape)
            imageio.imwrite(image_with_patches_path, image_with_patches)
        else:
            # print('else')
            image_path = os.path.join(raw_folder, name)
            output_image_path = os.path.join(JPGS_path, name.replace('.tif', '.jpg'))
            # print(image_path)
            base_image = imread(image_path, key=4)
            # print("Base", base_image.shape)
            # dim = (base_image.shape[1] // 2, base_image.shape[0] // 2)
            # base_image = cv2.resize(base_image, dim, interpolation=cv2.INTER_AREA)
            print(output_image_path, base_image.shape)
            imageio.imwrite(output_image_path, base_image)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.xml_path:
        og_path = args.image.split('/')[-1]
        bs_content = read_xml(args.xml_path)
        base_image = imread(args.image)
        print("Base", base_image.shape)
        # TODO: zrobić tak, żeby można było wybrać, który obrazek z xmla wybrać
        for image_xml in bs_content.find_all('image'):
            name = image_xml['name'].split('/')[-1].replace('.png', '.tif')

            if name == og_path:
                mask_image = get_mask(image_xml, base_image)
                patches = split_image(mask_image, 512)
                # TODO: zapisać wszystkie zdjęcia (wszystkie czy wszystkie z maską)
                patches = get_patches_with_mask(patches)
                patches = random_patches(patches, args.number_of_images)
                csv_out = [k[0] for k in patches]
                with open(args.gdrive + r'/patches.csv', 'w') as f:
                    json.dump(csv_out, f, indent=2)

                patches.sort(key=lambda x: x[0])
                print([k[0] for k in patches])
                patches, image_with_patches = draw_patches(patches, base_image)

                save_images(patches, args.out_dir)
                print(args.full_img, image_with_patches.shape)
                dim = (image_with_patches.shape[1] // 2, image_with_patches.shape[0] // 2)
                image_with_patches = cv2.resize(image_with_patches, dim, interpolation=cv2.INTER_AREA)
                print(args.full_img, image_with_patches.shape)
                imageio.imwrite(args.full_img, image_with_patches)

                # # FOR TESTING
                # if args.he_file:
                #     he_image = imread(args.he_file)
                #     he_image = cv2.rotate(he_image, cv2.ROTATE_180)
                #     for idx, batch in enumerate(patches):
                #         name, image = batch
                #
                #         p = [int(i) * 512 for i in name.split('_')[1:]]
                #         cv2.rectangle(he_image, (p[0] - 512, p[1] - 512), (p[0] + 512 + 512, p[1] + 512 + 512),
                #                       (0, 0, 0),
                #                       15)
                #     imageio.imwrite('/content/gdrive/MyDrive/raw/982-22-HE_copy2.jpg', he_image)

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
