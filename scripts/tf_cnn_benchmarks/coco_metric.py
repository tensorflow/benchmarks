# Copyright 2018 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""COCO-style evaluation metrics.

Forked from reference model implementation.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import json
import tempfile

from absl import flags

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import six

import tensorflow as tf

import ssd_constants

FLAGS = flags.FLAGS


# https://github.com/cocodataset/cocoapi/issues/49
if six.PY3:
  import pycocotools.coco
  pycocotools.coco.unicode = str


def compute_map(labels_and_predictions, val_json_file):
  """Use model predictions to compute mAP.

  The evaluation code is largely copied from the MLPerf reference
  implementation. While it is possible to write the evaluation as a tensor
  metric and use Estimator.evaluate(), this approach was selected for simplicity
  and ease of duck testing.
  """

  with tf.gfile.Open(val_json_file, "r") as f:
    annotation_data = json.load(f)

  predictions = []
  for example in labels_and_predictions:
    pred_box = example[ssd_constants.PRED_BOXES]
    pred_scores = example[ssd_constants.PRED_SCORES]

    loc, label, prob = decode_single(
        pred_box, pred_scores, ssd_constants.OVERLAP_CRITERIA,
        ssd_constants.MAX_NUM_EVAL_BOXES, ssd_constants.MAX_NUM_EVAL_BOXES)

    htot, wtot, _ = example[ssd_constants.RAW_SHAPE]
    for loc_, label_, prob_ in zip(loc, label, prob):
      # Ordering convention differs, hence [1], [0] rather than [0], [1]
      predictions.append([
          int(example[ssd_constants.SOURCE_ID]), loc_[1] * wtot, loc_[0] * htot,
          (loc_[3] - loc_[1]) * wtot, (loc_[2] - loc_[0]) * htot, prob_,
          ssd_constants.CLASS_INV_MAP[label_]
          ])

  if val_json_file.startswith("gs://"):
    _, local_val_json = tempfile.mkstemp(suffix=".json")
    tf.gfile.Remove(local_val_json)

    tf.gfile.Copy(val_json_file, local_val_json)
    atexit.register(tf.gfile.Remove, local_val_json)
  else:
    local_val_json = val_json_file

  cocoGt = COCO(local_val_json)
  cocoDt = cocoGt.loadRes(np.array(predictions))

  E = COCOeval(cocoGt, cocoDt, iouType='bbox')
  E.evaluate()
  E.accumulate()
  E.summarize()
  print("Current AP: {:.5f}".format(E.stats[0]))
  metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                  'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

  # Prefix with "COCO" to group in TensorBoard.
  return {"COCO/" + key: value for key, value in zip(metric_names, E.stats)}


def calc_iou(target, candidates):
  target_tiled = np.tile(target[np.newaxis, :], (candidates.shape[0], 1))
  # Left Top & Right Bottom
  lt = np.maximum(target_tiled[:,:2], candidates[:,:2])

  rb = np.minimum(target_tiled[:,2:], candidates[:,2:])

  delta = np.maximum(rb - lt, 0)

  intersect = delta[:,0] * delta[:,1]

  delta1 = target_tiled[:,2:] - candidates[:,:2]
  area1 = delta1[:,0] * delta1[:,1]
  delta2 = target_tiled[:,2:] - candidates[:,:2]
  area2 = delta2[:,0] * delta2[:,1]

  iou = intersect/(area1 + area2 - intersect)
  return iou


def decode_single(bboxes_in, scores_in, criteria, max_output, max_num=200):
  # Reference to https://github.com/amdegroot/ssd.pytorch

  bboxes_out = []
  scores_out = []
  labels_out = []

  for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
    score = np.squeeze(score, 1)

    # skip background
    if i == 0:
      continue

    mask = score > ssd_constants.MIN_SCORE
    if not np.any(mask):
      continue

    bboxes, score = bboxes_in[mask, :], score[mask]

    score_idx_sorted = np.argsort(score)
    score_sorted = score[score_idx_sorted]

    score_idx_sorted = score_idx_sorted[-max_num:]
    candidates = []

    # perform non-maximum suppression
    while len(score_idx_sorted):
      idx = score_idx_sorted[-1]
      bboxes_sorted = bboxes[score_idx_sorted, :]
      bboxes_idx = bboxes[idx, :]
      iou = calc_iou(bboxes_idx, bboxes_sorted)

      score_idx_sorted = score_idx_sorted[iou < criteria]
      candidates.append(idx)

    bboxes_out.append(bboxes[candidates, :])
    scores_out.append(score[candidates])
    labels_out.extend([i]*len(candidates))

  if len(scores_out) == 0:
    tf.logging.info("No objects detected. Returning dummy values.")
    return (
        np.zeros(shape=(1, 4), dtype=np.float32),
        np.zeros(shape=(1,), dtype=np.int32),
        np.ones(shape=(1,), dtype=np.float32) * ssd_constants.DUMMY_SCORE,
    )

  bboxes_out = np.concatenate(bboxes_out, axis=0)
  scores_out = np.concatenate(scores_out, axis=0)
  labels_out = np.array(labels_out)

  max_ids = np.argsort(scores_out)[-max_output:]

  return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]
