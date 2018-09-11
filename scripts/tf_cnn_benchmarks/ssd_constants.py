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
"""Central location for all constants related to MLPerf SSD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==============================================================================
# == Model =====================================================================
# ==============================================================================
IMAGE_SIZE = 300

# TODO(taylorrobie): MLPerf uses 80, but COCO documents 90. (RetinaNet uses 90)
# Update(taylorrobie): Labels > 81 show up in the pipeline. This will need to
#                      be resolved.
NUM_CLASSES = 91  # Including "no class"
NUM_SSD_BOXES = 8732

RESNET_DEPTH = 34

"""SSD specific"""
MIN_LEVEL = 3
MAX_LEVEL = 8

FEATURE_SIZES = (38, 19, 10, 5, 3, 1)
STEPS = (8, 16, 32, 64, 100, 300)

# https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
SCALES = (21, 45, 99, 153, 207, 261, 315)
ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
NUM_DEFAULTS = (4, 6, 6, 6, 4, 4)
NUM_DEFAULTS_BY_LEVEL = {3: 4, 4: 6, 5: 6, 6: 6, 7: 4, 8: 4}
SCALE_XY = 0.1
SCALE_HW = 0.2
BOX_CODER_SCALES = (1 / SCALE_XY, 1 / SCALE_XY, 1 / SCALE_HW, 1 / SCALE_HW)
MATCH_THRESHOLD = 0.5

# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

# SSD Cropping
NUM_CROP_PASSES = 50
CROP_MIN_IOU_CHOICES = (0, 0.1, 0.3, 0.5, 0.7, 0.9)
P_NO_CROP_PER_PASS = 1 / (len(CROP_MIN_IOU_CHOICES) + 1)

# Hard example mining
NEGS_PER_POSITIVE = 3


# ==============================================================================
# == Optimizer =================================================================
# ==============================================================================
LEARNING_RATE_SCHEDULE = (
    (0, 1e-3),
    (160000, 1e-4),
    (200000, 1e-5),
)
MOMENTUM = 0.9


# ==============================================================================
# == Keys ======================================================================
# ==============================================================================
BOXES = "boxes"
CLASSES = "classes"
NUM_BOXES = "num_boxes"
IMAGE = "image"
SOURCE_ID = "source_id"


# ==============================================================================
# == Evaluation ================================================================
# ==============================================================================

# Note: This is based on a batch size of 32
#   https://github.com/mlperf/reference/blob/master/single_stage_detector/ssd/train.py#L21-L37
CHECKPOINT_FREQUENCY = 20000
MAX_NUM_EVAL_BOXES = 200
