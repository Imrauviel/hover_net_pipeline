import copy
import cv2
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
          },
out = [{
    "version": 0,
    "tags": [],
    "shapes": [],
    "tracks": []
}]

parser = argparse.ArgumentParser(
    description='Generate project json file for CVAT')
parser.add_argument('dir_path', help='Name of the project')
parser.add_argument('-o', '-out_path', help='Name of the project', default='task_0/project.json')

if __name__ == '__main__':

    args = parser.parse_args()
    files = os.listdir(args.dir_path)
    for idx, file in enumerate(files):
        p = os.path.join(args.dir_path, file)
        mat_map = scipy.io.loadmat(p)
        bin_map = mat_map['inst_map']
        bin_map[bin_map != 0] = 1
        bin_map = bin_map.astype(np.uint8)
        contours, _ = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polygons = []
        for polygon in contours:
            coords = []
            for point in polygon:
                # a = (, )
                coords.append(point[0][0] / 1)
                coords.append(point[0][1] / 1)
            polygons.append(coords)
        temp = copy.copy(example)[0]
        temp['points'] = polygons
        out[0]['shapes'].append(temp)

    with open(args.out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Done.')
