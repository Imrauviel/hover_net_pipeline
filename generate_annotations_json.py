import argparse
import copy
import cv2
import json
import numpy as np
import os
import scipy.io

example = {
              "type": "polygon",
              "occluded": False,
              "outside": False,
              "z_order": 1,
              "rotation": 0.0,
              "points": None,
              "frame": 0,
              "group": 0,
              "source": "manual",
              "attributes": [],
              "elements": [],
              "label": "nowt. Ki67-"
          }

out = [{
    "version": 0,
    "tags": [],
    "shapes": [],
    "tracks": []
}]


def execute(path, project):
    """
    Args:
        path: path from where annotation will be loaded (should be folder with .mat files)
        project: structure, where annotations will be saved

    Returns:
        structure, with annotations in CVAT format

    1. We sort files(like in CVAT)
    2. For each lables file:
        2.1. We load that .mat file
        2.2. We convert it to binary mask (hovernet generets alot of different labels, but we need only one)
        2.3. We find contours in that mask
        2.4. For each contour:
            2.4.1. We create list of x, y coordinates of this contour [x1, y1, x2, y2, ...]
            2.4.2. We create copy of example
            2.4.3. We add coordinates to copy
            2.4.4. We add copy to output structure

    """
    files = sorted(os.listdir(args.dir_path))
    for idx, file in enumerate(files):
        mat_map = scipy.io.loadmat(os.path.join(path, file))
        bin_map = mat_map['inst_map']
        bin_map[bin_map != 0] = 1
        bin_map = bin_map.astype(np.uint8)
        contours, _ = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for polygon in contours:
            coords = []
            for point in polygon:
                coords.append(point[0][0] / 1)
                coords.append(point[0][1] / 1)
            temp = copy.deepcopy(example)
            temp['points'] = coords
            temp['frame'] = idx
            project[0]['shapes'].append(temp)
    return project


parser = argparse.ArgumentParser(
    description='Generate project json file for CVAT')
parser.add_argument('dir_path', help='Name of the project')
parser.add_argument('-o', '--out_path', help='Where save json with project', default='task_0/annotations.json')

if __name__ == '__main__':
    args = parser.parse_args()
    out = execute(args.dir_path, out)

    with open(args.out_path, 'w') as f:
        json.dump(out, f, indent=2)
