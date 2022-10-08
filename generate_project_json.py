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
if __name__ == '__main__':
    args = parser.parse_args()
    d['name'] = args.name
    with open('project.json', 'w') as f:
        json.dump(project, f)

    with open('task_0.json', 'w') as f:
        json.dump(task, f)