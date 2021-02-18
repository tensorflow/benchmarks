# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Simple script to diff benchmark results.

This script will read all the summary files from a base output directory and
print a human readable diff report.
"""

import json
import os
import sys


def _find_perfzero_logs(docker_output_dir):
  """Finds pairs of json_file, output_log file from all methods."""
  summary_files = []
  for root, _, files in os.walk(docker_output_dir):
    for summary_file in files:
      if summary_file.endswith('perfzero_summary.json'):
        full_summary_file = os.path.join(root, summary_file)
        summary_files.append(full_summary_file)
        sys.stdout.write('Found json {}\n'.format(full_summary_file))
  return summary_files


def _load_summaries(summary_files):
  """Loads input json file paths and returns json objects."""
  summary_jsons = []
  for summary_file in summary_files:
    with open(summary_file, 'r') as f:
      summary_json = json.load(f)
      summary_jsons.append(summary_json)
  return summary_jsons


def _summarize_benchmarks(summary_files):
  """Remaps list of json files -> summaries by benchmark method."""
  summary_jsons = _load_summaries(summary_files)
  performance_by_method = {}

  for summary_json in summary_jsons:
    method = summary_json['benchmark_result']['name']
    trial = summary_json['benchmark_result']['trial_id']
    metrics_list = summary_json['benchmark_result']['metrics']
    metrics = {}
    for metric_info in metrics_list:
      metrics[metric_info['name']] = metric_info['value']
    metrics['wall_time'] = summary_json['benchmark_result']['wall_time']
    label = summary_json['benchmark_info']['execution_label']

    performance_by_method.setdefault(method, {}).setdefault(label, [])
    performance_by_method[method][label].append((trial, metrics))

  return performance_by_method


def _print_diff_report(performance_by_method):
  """Prints a diff report of benchmark performance."""
  print('Performance report:')
  print(json.dumps(performance_by_method, indent=2))

  method_to_metric_to_perf = {}
  for method in performance_by_method:
    for label, label_data in performance_by_method[method].items():
      latest_trial_data = max(label_data, key=lambda x: x[0])
      latest_metrics = latest_trial_data[1]
      for metric, value in latest_metrics.items():
        method_to_metric_to_perf.setdefault(method, {}).setdefault(metric, [])
        method_to_metric_to_perf[method][metric].append((label, value))

  print('Diff report:')
  for method in sorted(method_to_metric_to_perf):
    print('-- benchmark: {}'.format(method))
    for metric in sorted(method_to_metric_to_perf[method].keys()):
      value_list = []
      for label, value in sorted(
          method_to_metric_to_perf[method][metric], key=lambda x: x[0]):
        print('         {}: {}: {}'.format(metric, label, value))
        value_list.append(value)

      if len(value_list) == 2:
        control_val = float(value_list[0])
        expt_val = float(value_list[1])
        if abs(control_val) > 1e-5:
          diff_pct = (expt_val / control_val - 1.0) * 100.0
        else:
          diff_pct = -1.0
        print('             diff: {:2.2f}%'.format(diff_pct))


def main():
  if len(sys.argv) != 2:
    raise RuntimeError('Usage: {} <base perfzero output dir>'.format(
        sys.argv[0]))

  perfzero_output_dir = sys.argv[1]
  summary_files = _find_perfzero_logs(perfzero_output_dir)
  performance_by_method = _summarize_benchmarks(summary_files)
  _print_diff_report(performance_by_method)


if __name__ == '__main__':
  main()
