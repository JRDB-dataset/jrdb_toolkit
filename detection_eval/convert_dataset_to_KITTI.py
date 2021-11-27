import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import shutil
import numpy as np
import open3d as o3d
import yaml


IN_IMG_PATH = 'images/image_stitched/%s/%s.jpg'
IN_PTC_LOWER_PATH = 'pointclouds/lower_velodyne/%s/%s.pcd'
IN_PTC_UPPER_PATH = 'pointclouds/upper_velodyne/%s/%s.pcd'
IN_LABELS_3D = 'processed_labels/labels_3d/*.json'
IN_LABELS_2D = 'processed_labels/labels_2d_stitched/*.json'
IN_DETECTIONS_2D = 'detections/*.json'
IN_CALIBRATION_F = 'calibration/defaults.yaml'

OUT_IMG_PATH = 'image_2'
OUT_PTC_PATH = 'velodyne'
OUT_LABEL_PATH = 'label_2'
OUT_DETECTION_PATH = 'detection'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o',
                    '--output_dir',
                    default='KITTI',
                    help='location to store dataset in KITTI format')
    ap.add_argument('-i',
                    '--input_dir',
                    default='dataset',
                    help='root directory in jrdb format')
    return ap.parse_args()


def get_file_list(input_dir):
    def _filepath2filelist(path):
        return set(
                tuple(os.path.splitext(f)[0].split(os.sep)[-2:])
                for f in glob.glob(os.path.join(input_dir, path % ('*', '*'))))

    def _label2filelist(path, key='labels'):
        seq_dicts = []
        for json_f in glob.glob(os.path.join(input_dir, path)):
            with open(json_f) as f:
                labels = json.load(f)
            seq_name = os.path.basename(os.path.splitext(json_f)[0])
            seq_dicts.append({(seq_name, os.path.splitext(file_name)[0]): label
                              for file_name, label in labels[key].items()
                              })
        return dict(collections.ChainMap(*seq_dicts))

    imgs = _filepath2filelist(IN_IMG_PATH)
    lower_ptcs = _filepath2filelist(IN_PTC_LOWER_PATH)
    upper_ptcs = _filepath2filelist(IN_PTC_UPPER_PATH)
    labels_2d = _label2filelist(IN_LABELS_2D)
    labels_3d = _label2filelist(IN_LABELS_3D)
    detections_2d = _label2filelist(IN_DETECTIONS_2D, key='detections')
    filelist = set.intersection(
            imgs, lower_ptcs, upper_ptcs, labels_2d.keys(), labels_3d.keys())

    return {f: (labels_2d[f], labels_3d[f], detections_2d[f])
            for f in sorted(filelist)}


def move_frame(input_dir, output_dir, calib, seq_name, file_name, labels_2d,
               labels_3d, detection_2d, file_idx):
    def _load_pointcloud(path, calib_key):
        ptc = np.asarray(o3d.io.read_point_cloud(
                os.path.join(input_dir, path % (seq_name, file_name))).points)
        ptc -= np.expand_dims(np.array(calib[calib_key]['translation']), 0)
        theta = -float(calib[calib_key]['rotation'][-1])
        rotation_matrix = np.array(
            ((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        ptc[:, :2] = np.squeeze(
            np.matmul(rotation_matrix, np.expand_dims(ptc[:, :2], 2)))
        return ptc

    # Copy image
    shutil.copy(os.path.join(input_dir, IN_IMG_PATH % (seq_name, file_name)),
                os.path.join(output_dir, OUT_IMG_PATH, f'{file_idx:06d}.jpg'))

    # Copy point cloud
    lower_ptc = _load_pointcloud(IN_PTC_LOWER_PATH, 'lidar_lower_to_rgb')
    upper_ptc = _load_pointcloud(IN_PTC_UPPER_PATH, 'lidar_upper_to_rgb')
    ptc = np.vstack((upper_ptc, lower_ptc))
    ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1))))
    np.save(os.path.join(output_dir, OUT_PTC_PATH, f'{file_idx:06d}.npy'), ptc)

    label_id_2d = set(f['label_id'] for f in labels_2d)
    label_id_3d = set(f['label_id'] for f in labels_3d)
    label_ids = label_id_2d.intersection(label_id_3d)

    # Create label
    label_lines = []
    for label_id in label_ids:
        if not label_id.startswith('pedestrian:'):
            continue
        label_2d = [l for l in labels_2d if l['label_id'] == label_id][0]
        label_3d = [l for l in labels_3d if l['label_id'] == label_id][0]
        rotation_y = (-label_3d['box']['rot_z']
                      if label_3d['box']['rot_z'] < np.pi else 2 * np.pi -
                      label_3d['box']['rot_z'])
        label_lines.append(
            f"Pedestrian 0 0 {label_3d['observation_angle']} "
            f"{label_2d['box'][0]} {label_2d['box'][1]} "
            f"{label_2d['box'][0] + label_2d['box'][2]} "
            f"{label_2d['box'][1] + label_2d['box'][3]} "
            f"{label_3d['box']['h']} {label_3d['box']['w']} "
            f"{label_3d['box']['l']} {-label_3d['box']['cy']} "
            f"{-label_3d['box']['cz'] + label_3d['box']['h'] / 2} "
            f"{label_3d['box']['cx']} {rotation_y} 1\n"
        )
    label_out = os.path.join(output_dir, OUT_LABEL_PATH, f'{file_idx:06d}.txt')
    with open(label_out, 'w') as f:
        f.writelines(label_lines)

    # Create detection
    detection_lines = []
    for detection in detection_2d:
        if not detection['label_id'].startswith('person:'):
            continue
        detection_lines.append(
            f"Pedestrian 0 0 -1 "
            f"{detection['box'][0]} {detection['box'][1]} "
            f"{detection['box'][0] + detection['box'][2]} "
            f"{detection['box'][1] + detection['box'][3]} "
            f"-1 -1 -1 -1 -1 -1 -1 {detection['score']}\n"
        )
    detection_out = os.path.join(
            output_dir, OUT_DETECTION_PATH, f'{file_idx:06d}.txt')
    with open(detection_out, 'w') as f:
        f.writelines(detection_lines)


def convert_jr2kitti(input_dir, output_dir, file_list):

    os.makedirs(os.path.join(output_dir, OUT_IMG_PATH))
    os.makedirs(os.path.join(output_dir, OUT_PTC_PATH))
    os.makedirs(os.path.join(output_dir, OUT_LABEL_PATH))
    os.makedirs(os.path.join(output_dir, OUT_DETECTION_PATH))

    with open(os.path.join(output_dir, 'filelist.txt'), 'w') as f:
        f.write('\n'.join(a + ' ' + b for a, b in file_list.keys()))

    pool = mp.Pool(20)
    with open(os.path.join(input_dir, IN_CALIBRATION_F)) as f:
        calib = yaml.safe_load(f)['calibrated']

    pool.starmap(
        move_frame,
        [(input_dir, output_dir, calib, seq_name, file_name, label_2d,
          label_3d, detection_2d, idx)
         for idx, ((seq_name, file_name),
                   (label_2d, label_3d, detection_2d))
         in enumerate(file_list.items())])
    shutil.copytree(os.path.join(input_dir, 'calibration'),
                    os.path.join(output_dir, 'calib'))


if __name__ == "__main__":
    args = parse_args()
    file_list = get_file_list(args.input_dir)
    convert_jr2kitti(args.input_dir, args.output_dir, file_list)
