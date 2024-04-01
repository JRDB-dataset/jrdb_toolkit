# Evaluation toolkit for JRDB-PanoTrack

This repo contain code for [JRDB dataset and benchmark](https://jrdb.erc.monash.edu/panotrack)

Note that the predictions must follow the format of the ground truth labels. The predictions must be in the form of a json file with the following format:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "road"
    },
    {
      "id": 2,
      "name": "terrain"
    },
    ...
    ],
    
    "images": [
        {
            ...
        }
    ],
  "annotations": [
    {
      "image_id": 0,
      "category_id": 1,
      "segmentation": {
        "counts": ... RLE encoded mask ...,
        "size": [height, width]
      },
      "tracking_id": 0 # for tracking evaluation only
    },
    ...
  ]
}
```

For Closed-world and open-world panoptic segmentation evaluation, run:

```bash
    python panoptic_eval/run_jrdb_panoseg.py
    --TRACKERS_FOLDER /path/to/predictions_2d_panoptic_CW
    --GT_FOLDER /path/to/labels_2d_panoptic_CW
    --split val 
    --eval_OW False # False for closed-world evaluation, True for open-world evaluation
    --eval_stitched False # False for individual frames, True for stitched frames
    --run_parallel False # False for single process, True for parallel processing
```