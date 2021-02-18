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
  print('Diff report:')
  print(json.dumps(performance_by_method, indent=2))


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
