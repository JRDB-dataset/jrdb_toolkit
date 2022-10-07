#!/bin/bash

GT_FOLDER="${1:-/home/tho/GoogleDrive/code/jdrb_toolkit_official/pose_eval/labels_2d_pose_stitched_coco}"
TRACKERS_FOLDER="${2:-/home/tho/GoogleDrive/code/jdrb_toolkit_official/pose_eval/labels_2d_pose_stitched_coco}"

python3 scripts/run_posetrack_challenge.py --GT_FOLDER $GT_FOLDER --TRACKERS_FOLDER $TRACKERS_FOLDER \
       --USE_PARALLEL True \
       --NUM_PARALLEL_CORES 8
