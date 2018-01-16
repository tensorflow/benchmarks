# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utility code for the default platform."""

import cnn_util


def get_platform_params():
  """Returns a dict of platform-specific params.

  No platform-specific flags are needed for the default platform, so this
  returns an empty dict.

  Returns:
    A dict that maps from param name to ParamSpec.
  """
  return {}


def get_cluster_manager(params, config_proto):
  """Returns the cluster manager to be used."""
  return cnn_util.GrpcClusterManager(params, config_proto)


def _initialize(params, config_proto):
  # Currently, no platform initialization needs to be done.
  del params, config_proto


_is_initalized = False


def initialize(params, config_proto):
  global _is_initalized
  if _is_initalized:
    return
  _is_initalized = True
  _initialize(params, config_proto)
