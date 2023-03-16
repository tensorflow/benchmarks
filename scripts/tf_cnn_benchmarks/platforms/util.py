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

"""Utility code for a certain platform.

This file simply imports everything from the default platform. To switch to a
different platform, the import statement can be changed to point to a new
platform.

Creating a custom platform can be useful to, e.g., run some initialization code
required by the platform or register a platform-specific model.
"""

from platforms.default.util import *  # pylint: disable=unused-import,wildcard-import
