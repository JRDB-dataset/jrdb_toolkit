import argparse
import collections
import glob
import json
import os

import numpy as np

BASE_PATH = "/pvol2/jrdb_dev/jrdb_website_dev/static/downloads/jrdb_train/train_dataset_with_activity/"

INPUT_3D_LABELS_PATH = BASE_PATH + "labels/labels_3d/*.json"
INPUT_2D_LABELS_PATH = BASE_PATH + "labels/labels_2d_stitched/*.json"
LABEL_ROOT_KEY = "labels"
ENUM_OCCLUSION = (
    "Fully_visible",
    "Mostly_visible",
    "Severely_occluded",
    "Fully_occluded",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output_kitti_dir",
        default="KITTI",
        help="location of the output KITTI-like labels",
    )
    ap.add_argument(
        "-i",
        "--input_jrdb_dir",
        default="test_dataset/labels",
        help="location of the input jrdb labels",
    )
    return ap.parse_args()


def get_labels(input_dir,path):
    """Read label directory

        Args:
            input_dir (str): Input directory of the jrdb labels.

        Returns:
            dict: {(seq_name, seq_idx) -> ([labels_2d, ...], [labels_3d, ...])}
    """

    """Read label path of 2D/3D labels

    Args:
        path (str): Input path of the jrdb labels.

    Returns:
        dict: {(seq_name, seq_idx) -> [labels, ...]}
    """
    seq_dicts = []
    output_dir=args.output_kitti_dir
    for seq_index,json_f in enumerate(glob.glob(os.path.join(input_dir, path))):
        seq_name=json_f.split('/')[-1].split('.')[0]
        with open(json_f) as f:
            labels_2d = json.load(f)
        with open(json_f.replace("labels_2d_stitched", "labels_3d")) as f:
            labels_3d = json.load(f)
        lines=[]
        for i, (k, v2) in enumerate(labels_2d["labels"].items()):
            frame = int(k.split(".")[0])
            (k3, v3) = list(labels_3d["labels"].items())[i]
            assert int(k3.split(".")[0]) == int(k.split(".")[0])
            pedestrians = {}
            for obj in v2:
                track_id = obj["label_id"]
                if track_id not in pedestrians:
                    pedestrians[track_id] = [obj, None]
                else:
                    pedestrians[track_id][0] = obj
            for obj in v3:
                track_id = obj["label_id"]
                if track_id not in pedestrians:
                    pedestrians[track_id] = [None, obj]
                else:
                    pedestrians[track_id][1] = obj
            for k, (v2d, v3d) in pedestrians.items():
                if v2d != None:
                    attributes_2d = v2d["attributes"]
                    track_id = v2d["label_id"]
                    truncated = 0
                    occlusion = ENUM_OCCLUSION.index(attributes_2d["occlusion"])
                    x1_2d = v2d["box"][0]
                    y1_2d = v2d["box"][1]
                    x2_2d = v2d["box"][0] + v2d["box"][2]
                    y2_2d = v2d["box"][1] + v2d["box"][3]
                else:
                    truncated, occlusion, x1_2d, y1_2d, x2_2d, y2_2d = (
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                    )
                if v3d != None:
                    rotation_y = (
                        -v3d["box"]["rot_z"]
                        if v3d["box"]["rot_z"] < np.pi
                        else 2 * np.pi - v3d["box"]["rot_z"]
                    )
                    alpha = v3d["observation_angle"]
                    height_3d = v3d["box"]["h"]
                    width_3d = v3d["box"]["w"]
                    length_3d = v3d["box"]["l"]
                    centerx_3d = -v3d["box"]["cy"]
                    centery_3d = -v3d["box"]["cz"] + v3d["box"]["h"] / 2
                    centerz_3d = v3d["box"]["cx"]
                else:
                    (
                        rotation_y,
                        alpha,
                        height_3d,
                        width_3d,
                        length_3d,
                        centerx_3d,
                        centery_3d,
                        centerz_3d,
                    ) = (-1, -1, -1, -1, -1, -1, -1, -1)
                line = (
                    "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n"
                    % (
                        frame,
                        k.split(':')[-1],
                        truncated,
                        occlusion,
                        alpha,
                        x1_2d,
                        y1_2d,
                        x2_2d,
                        y2_2d,
                        height_3d,
                        width_3d,
                        length_3d,
                        centerx_3d,
                        centery_3d,
                        centerz_3d,
                        rotation_y,
                    )
                )
                lines.append(line)
        seq_dir = os.path.join(output_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        with open(os.path.join(seq_dir, str(seq_index) + ".txt"), "w") as f:
            f.writelines(lines)
    return lines


def convert_jr2kitti(labels, output_dir):
    """Write jrdb labels to output_dir in KITTI-like format of text file.

    Args:
        labels (dict): {(seq_name, seq_idx) ->
                        ([labels_2d, ...], [labels_3d, ...])}
        output_dir (str): Output directory of the converted label.
    """

    def _label_key(label):
        return label["label_id"]

    # Parse all sequences of the given label.
    for (seq_name, seq_idx), (labels_2d, labels_3d) in labels.items():
        # Join 2D/3D labels based on the given label key.
        labels_2d = {_label_key(label): label for label in labels_2d}
        labels_3d = {_label_key(label): label for label in labels_3d}
        label_all = {
            k: (labels_2d.get(k), labels_3d.get(k))
            for k in set(labels_2d).union(labels_3d)
        }
        # Parse each pedestrian in a given sequence.
        label_lines = []
        for i, (label_2d, label_3d) in enumerate(label_all.values()):
            # Sanity check.
            if label_2d is not None and label_3d is not None:
                assert _label_key(label_2d) == _label_key(label_3d)
            assert not (label_2d is None and label_3d is None)

            # Ignore all labels else than pedestrian.
            if not _label_key(label_2d or label_3d).startswith("pedestrian:"):
                continue

            # Initialize all label attributes
            rotation_y, num_points_3d, alpha, height_3d = -1, -1, -1, -1
            width_3d, length_3d, centerx_3d, centery_3d = -1, -1, -1, -1
            centerz_3d, x1_2d, y1_2d, x2_2d, y2_2d = -1, -1, -1, -1, -1
            truncated, occlusion = -1, -1

            # Fill in values extracted from 2D label.
            if label_2d is not None:
                x1_2d = label_2d["box"][0]
                y1_2d = label_2d["box"][1]
                x2_2d = label_2d["box"][0] + label_2d["box"][2]
                y2_2d = label_2d["box"][1] + label_2d["box"][3]
                attributes_2d = label_2d["attributes"]
                truncated = int(attributes_2d["truncated"].lower() == "true")
                occlusion = ENUM_OCCLUSION.index(attributes_2d["occlusion"])

            # Fill in values extracted from 3D label.
            if label_3d is not None:
                rotation_y = (
                    -label_3d["box"]["rot_z"]
                    if label_3d["box"]["rot_z"] < np.pi
                    else 2 * np.pi - label_3d["box"]["rot_z"]
                )
                attributes_3d = label_3d["attributes"]
                num_points_3d = attributes_3d["num_points"]
                alpha = label_3d["observation_angle"]
                height_3d = label_3d["box"]["h"]
                width_3d = label_3d["box"]["w"]
                length_3d = label_3d["box"]["l"]
                centerx_3d = -label_3d["box"]["cy"]
                centery_3d = -label_3d["box"]["cz"] + label_3d["box"]["h"] / 2
                centerz_3d = label_3d["box"]["cx"]

            # Append a line of text in a KITTI-like format.
            label_lines.append(
                "Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s 1\n"
                % (
                    truncated,
                    occlusion,
                    num_points_3d,
                    alpha,
                    x1_2d,
                    y1_2d,
                    x2_2d,
                    y2_2d,
                    height_3d,
                    width_3d,
                    length_3d,
                    centerx_3d,
                    centery_3d,
                    centerz_3d,
                    rotation_y,
                )
            )
        # Write label text file to the output directory.
        seq_dir = os.path.join(output_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        with open(os.path.join(seq_dir, str(seq_idx) + ".txt"), "w") as f:
            f.writelines(label_lines)


if __name__ == "__main__":
    args = parse_args()
    labels = get_labels(BASE_PATH,INPUT_2D_LABELS_PATH)

