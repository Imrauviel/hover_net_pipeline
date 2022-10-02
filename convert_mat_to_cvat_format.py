import argparse

import xmltodict
import scipy.io
import cv2
import numpy as np
import os
from typing import List, Dict

base_json = {
    'annotations': {
        'version': 1.1,
        'meta': {
            'task': {
                'id': 2,
                'name': 'cell_task',
                'size': 1,
                'mode': 'annotation',
                'overlap': 0,
                'bugtracker': None,
                'flipped': False,
                'created': "2018-09-25 11:34:24.617558+03:00",
                'updated': "2018-09-25 11:38:27.301183+03:00",
                'labels': [
                    {
                        'label': [
                            {
                                'name': 'nowt. Ki67-',
                                'color': '#2212D1',
                                'attributes': None
                            },
{
                                'name': 'nowt. Ki67+',
                                'color': '#FF002C',
                                'attributes': None
                            },
{
                                'name': 'Inne',
                                'color': '#05694B',
                                'attributes': None
                            }
                        ]
                    },
                ],
                'owner': {
                    'username': 'admin',
                    'email': 'admin@admin.pl'
                }
            },
            'dumped': '2018-09-25 11:38:28.799808+03:00'
        },
    }
}


def save_xml(file_name: str, file: Dict) -> None:
    with open(f'{file_name}', 'w') as result_file:
        result_file.write(xmltodict.unparse(file, pretty=True))


def polygon_to_str(polygon):
    return ';'.join([(str(float(a[0]))) + ',' + str(float(a[1])) for a in polygon])


def mat_to_polygon(mat_path):
    mat_map = scipy.io.loadmat(mat_path)
    bin_map = mat_map['inst_map']
    bin_map[bin_map != 0] = 1
    bin_map = bin_map.astype(np.uint8)
    contours, _ = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for polygon in contours:
        coords = []
        for point in polygon:
            a = (int(point[0][0]), int(point[0][1]))
            coords.append(a)
        polygons.append(coords)
    return polygons


def execute(args):
    images = []
    files = os.listdir(args.dir_path)
    for idx, file in enumerate(files):
        polygons = mat_to_polygon(os.path.join(args.dir_path, file))
        polygon_dicts: List[Dict] = []
        for polygon in polygons:
            new_polygon = {
                "@label": 'nowt. Ki67-',
                "@points": str(polygon_to_str(polygon)),
                "@occluded": 0,
                "@z_order": 1
            }
            polygon_dicts.append(new_polygon)
        image_dict = {
            "@id": idx,
            "@name": file.replace('.mat', '.png'),
            "@width": args.width,
            "@height": args.height,
            "polygon": polygon_dicts
        }
        images.append(image_dict)
    base_json['annotations']['image'] = images
    base_json['annotations']['meta']['task']['size'] = len(files)
    save_xml(args.out_name, base_json)
    return


parser = argparse.ArgumentParser(
    description='Convert .mat polygons to CVAT1.1 format.')
parser.add_argument('dir_path', help='Path to dir with mat files.')
parser.add_argument('-o', '--out-name', default='annotations.xml',
                    help='Output name file.. Default: annotations.xml')
parser.add_argument('-w', '--width', default=512, type=int,
                    help='Width of image. Default: 512')
parser.add_argument('-he', '--height', default=512, type=int,
                    help="Height of image. Default: 512")
if __name__ == '__main__':
    args = parser.parse_args()
    execute(args)
    print(f"Annotations save as {args.out_name}.")
