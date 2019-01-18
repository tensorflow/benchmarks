# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Bench result object."""


class BenchResult(object):
  """Holds results of a benchmark test."""

  def __init__(self):
    self.total_time = None
    self.top_1 = None
    self.top_5 = None
    self.exp_per_second = None
    self.other_results = []

  def add_top_1(self, result, expected_min=None, expected_max=None):
    """Adds a top_1 result entry."""
    self.add_other_quality(
        result, 'top_1', expected_min=expected_min, expected_max=expected_max)

  def add_top_5(self, result, expected_min=None, expected_max=None):
    """Adds a top_5 result entry."""
    self.add_other_quality(
        result, 'top_5', expected_min=expected_min, expected_max=expected_max)

  def add_examples_per_second(self, result):
    """Adds examples per second result entry."""
    self.add_result(result, 'exp_per_second', 'exp_per_second')

  def add_other_quality(self,
                        result,
                        quality_unit,
                        expected_min=None,
                        expected_max=None):
    """Adds a quality entry to the results list.

    Args:
      result: result to be recorded
      quality_unit: type of quality result, e.g. top_1
      expected_min: minimum expected value to determine pass/fail of result.
      expected_max: max expected value to determine pass/fail of result.
    """
    result_status = None
    if expected_min or expected_max:
      result_status = 'FAILED'
      if expected_min and expected_max:
        if result >= expected_min and result <= expected_max:
          result_status = 'PASS'
      elif expected_min:
        if result >= expected_min:
          result_status = 'PASS'
      elif expected_max:
        if result <= expected_max:
          result_status = 'PASS'

    self._add_result(
        self.other_results,
        result,
        'quality',
        quality_unit,
        expected_min=expected_min,
        expected_max=expected_max,
        result_status=result_status)

  def add_result(self, result, result_type, result_unit):
    """Add benchmark result to list."""
    self._add_result(self.other_results, result, result_type, result_unit)

  def get_results(self):
    """Returns top_1 result entry or none if not set."""
    result_list = []
    if self.top_1:
      self._add_result(result_list, self.top_1, 'quality', 'top_1')
    if self.top_5:
      self._add_result(result_list, self.top_5, 'quality', 'top_5')
    if self.exp_per_second:
      self._add_result(result_list, self.exp_per_second, 'exp_per_second',
                       'exp_per_second')
    result_list.extend(self.other_results)

    return result_list

  def _add_result(self,
                  result_list,
                  result,
                  result_type,
                  result_unit,
                  expected_min=None,
                  expected_max=None,
                  result_status=None):
    """Adds result to result list passed."""
    if result:
      result_dict = {}
      result_dict['result'] = result
      result_dict['result_type'] = result_type
      result_dict['result_unit'] = result_unit
      if expected_min:
        result_dict['expected_min'] = expected_min
      if expected_max:
        result_dict['expected_max'] = expected_max
      if result_status:
        result_dict['result_status'] = result_status
      result_list.append(result_dict)
