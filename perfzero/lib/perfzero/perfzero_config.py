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
"""PerfZero configs provided by user."""
from __future__ import print_function

import os


def add_parser_arguments(parser):
  """Add arguments to the parser intance."""
  parser.add_argument(
      '--force_update',
      action='store_true')
  parser.add_argument(
      '--gce_nvme_raid',
      type=str,
      default=None,
      help='If set create raid 0 array with devices at disk_dir')
  parser.add_argument(
      '--gcs_downloads',
      type=str,
      default=None,
      help='List of GCS urls separated by \',\' to download data')
  parser.add_argument(
      '--git_repos',
      type=str,
      default=None,
      help='List of git repository url separated by \',\' to be checked-out')
  parser.add_argument(
      '--benchmark_methods',
      action='append',
      type=str,
      default=[])
  parser.add_argument(
      '--ml_framework_build_label',
      type=str,
      default=None)
  parser.add_argument(
      '--execution_label',
      type=str,
      default=None)
  parser.add_argument(
      '--platform_name',
      type=str,
      default=None)
  parser.add_argument(
      '--system_name',
      type=str,
      default=None)
  parser.add_argument(
      '--output_gcs_url',
      type=str,
      default=None)
  parser.add_argument(
      '--bigquery_project_name',
      type=str,
      default=None)
  parser.add_argument(
      '--bigquery_dataset_table_name',
      type=str,
      default=None)
  parser.add_argument(
      '--python_path',
      type=str,
      default=None)
  parser.add_argument(
      '--workspace',
      type=str,
      default='workspace')
  parser.add_argument(
      '--dockerfile_path',
      type=str,
      default='docker/Dockerfile',
      help='Path to docker file')


class PerfZeroConfig(object):
  """Creates and contains config for PerfZero."""

  def __init__(self, mode, flags=None):
    self.mode = mode
    self.flags = flags
    if mode == 'flags':
      self.gce_nvme_raid_str = flags.gce_nvme_raid
      self.gcs_downloads_str = flags.gcs_downloads
      self.git_repos_str = flags.git_repos
      self.benchmark_method_patterns = flags.benchmark_methods
      self.ml_framework_build_label_str = flags.ml_framework_build_label
      self.execution_label_str = flags.execution_label
      self.platform_name_str = flags.platform_name
      self.system_name_str = flags.system_name
      self.output_gcs_url_str = flags.output_gcs_url
      self.bigquery_project_name_str = flags.bigquery_project_name
      self.bigquery_dataset_table_name_str = flags.bigquery_dataset_table_name
      self.output_gcs_url_str = flags.output_gcs_url
      self.python_path_str = flags.python_path
      self.workspace = flags.workspace
      self.force_update = flags.force_update
      self.dockerfile_path = flags.dockerfile_path

  def get_env_vars(self):
    env_vars = {}
    for key in os.environ.keys():
      if key.startswith('PERFZERO_'):
        env_vars[key] = os.environ[key]
    return env_vars

  def get_flags(self):
    not_none_flags = {}
    for key in vars(self.flags):
      value = getattr(self.flags, key)
      if value is not None:
        not_none_flags[key] = value
    return not_none_flags

  def get_git_repos(self, site_packages_dir):
    """Parse git repository string."""
    git_repos = []
    if not self.git_repos_str:
      return git_repos

    for repo_entry in self.git_repos_str.split(','):
      parts = repo_entry.split(';')
      git_repo = {}
      if len(parts) == 1:
        # Assume the git url has format */{dir_name}.git
        git_repo['dir_name'] = parts[0].rsplit('/', 1)[-1].rsplit('.', 1)[0]
        git_repo['url'] = parts[0]
      if len(parts) >= 2:
        git_repo['dir_name'] = parts[0]
        git_repo['url'] = parts[1]
      git_repo['local_path'] = os.path.join(site_packages_dir,
                                            git_repo['dir_name'])
      if len(parts) >= 3:
        git_repo['branch'] = parts[2]
      if len(parts) >= 4:
        git_repo['git_hash'] = parts[3]
      git_repos.append(git_repo)

    return git_repos

  def get_gcs_downloads(self, data_dir):
    """Download data from GCS."""
    gcs_downloads = []
    if not self.gcs_downloads_str:
      return gcs_downloads

    for entry in self.gcs_downloads_str.split(','):
      gcs_download = {}
      # Canonicalize gcs url to remove trailing '/' and '*'
      if entry.endswith('*'):
        entry = entry[:-1]
      if entry.endswith('/'):
        entry = entry[:-1]

      if ';' in entry:
        gcs_download['gcs_url'] = entry.split(';')[1]
        gcs_download['local_path'] = os.path.join(data_dir, entry.split(';')[0])
      else:
        gcs_download['gcs_url'] = entry
        gcs_download['local_path'] = os.path.join(data_dir, os.path.basename(entry))
    gcs_downloads.append(gcs_download)

    return gcs_downloads

