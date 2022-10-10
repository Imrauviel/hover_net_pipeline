import argparse
import json

project = {
    "name": None,
    "labels": [
        {
            "name": "Inne",
            "color": "#05694b",
            "attributes": [],
            "type": "any",
            "sublabels": []
        },
        {
            "name": "nowt. Ki67+",
            "color": "#ff002c",
            "attributes": [],
            "type": "any",
            "sublabels": []
        },
        {
            "name": "nowt. Ki67-",
            "color": "#2212d1",
            "attributes": [],
            "type": "any",
            "sublabels": []
        }
    ],
    "bug_tracker": "",
    "status": "annotation",
    "version": "1.0"
}

task = {
    "name": "default",
    "bug_tracker": "",
    "status": "annotation",
    "labels": [
        {
            "name": "Inne",
            "color": "#05694b",
            "attributes": [],
            "type": "any",
            "sublabels": []
        },
        {
            "name": "nowt. Ki67+",
            "color": "#ff002c",
            "attributes": [],
            "type": "any",
            "sublabels": []
        },
        {
            "name": "nowt. Ki67-",
            "color": "#2212d1",
            "attributes": [],
            "type": "any",
            "sublabels": []
        }
    ],
    "subset": "default",
    "version": "1.0",
    "data": {
        "chunk_size": 72,
        "image_quality": 90,
        "start_frame": 0,
        "stop_frame": 19,
        "storage_method": "file_system",
        "storage": "local",
        "sorting_method": "lexicographical",
        "chunk_type": "imageset",
        "deleted_frames": []
    },
    "jobs": [
        {
            "start_frame": 0,
            "stop_frame": 19,
            "status": "annotation"
        }
    ]
}

parser = argparse.ArgumentParser(
    description='Generate project json file for CVAT')
parser.add_argument('name', help='Name of the project')
parser.add_argument('-p', '--project_path', help='path to project.json', default='backup/project.json')
parser.add_argument('-t', '--task_path', help='path to task.json', default='backup/task_0/task.json')
parser.add_argument('-i', '--experiment_id',
                    help='Id of experiment. Usefully if we have a lot of experiments from same image. ', default='0')

if __name__ == '__main__':
    args = parser.parse_args()
    project['name'] = args.name + '_' + args.experiment_id

    with open(args.project_path, 'w') as f:
        json.dump(project, f)

    with open(args.task_path, 'w') as f:
        json.dump(task, f)
