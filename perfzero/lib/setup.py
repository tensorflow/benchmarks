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

"""Checkout repository, download data and build docker image."""
from __future__ import print_function

import argparse
import os

import perfzero.common.device_utils as device_utils
import perfzero.common.utils as utils

parser = argparse.ArgumentParser()


class SetupRunner(object):
  """Checkout repository, download data and build docker image."""

  def __init__(self,
               docker_file,
               docker_tag='temp/tf-gpu',
               gce_nvme_raid=None,
               data_dir='/data'):
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_dir = os.path.join(project_dir, 'workspace')
    self.docker_file_path = os.path.join(project_dir, docker_file)
    self.auth_token_dir = os.path.join(workspace_dir, 'auth_tokens')
    self.site_packages_dir = os.path.join(workspace_dir, 'site-packages')

    self.docker_tag = docker_tag
    self.gce_nvme_raid = gce_nvme_raid
    self.data_dir = data_dir

  def setup(self):
    """Builds and runs docker image with specified test config."""
    utils.check_and_print_env_var()

    # Download gcloud auth token.
    utils.download_from_gcs('gs://tf-performance/auth_tokens/*',
                            self.auth_token_dir)

    # Set up the raid array.
    if self.gce_nvme_raid == 'all':
      devices = device_utils.get_nvme_devices()
      device_utils.create_gce_nvme_raid(self.data_dir, devices)

    # Check out git repos
    git_repos = self._get_git_repos()
    for git_repo in git_repos:
      utils.checkout_git_repo(
          git_repo.get('url'),
          os.path.join(self.site_packages_dir, git_repo.get('local_path')),
          branch=git_repo.get('branch'),
          sha_hash=git_repo.get('sha_hash'))

    # Download data
    gcs_downloads = self._get_gcs_downloads()
    for gcs_download in gcs_downloads:
      local_path = os.path.join(self.data_dir, gcs_download['local_path'])
      utils.download_from_gcs(gcs_download['gcs_path'], local_path)

    # Build docker image.
    docker_build_cmd = 'docker build --pull -f {} -t {} .'.format(
        self.docker_file_path, self.docker_tag)
    utils.run_commands([docker_build_cmd])

  def _get_gcs_downloads(self):
    """Return list of gcs locations to download."""
    gcs_str = utils.get_env_var('ROGUE_GCS_DOWNLOADS')
    gcs_downloads = []
    for gcs_entry in gcs_str.split(','):
      gcs_parts = gcs_entry.split(';')
      gcs_download = {}
      gcs_download['local_path'] = gcs_parts[0]
      gcs_download['gcs_path'] = gcs_parts[1]
      gcs_downloads.append(gcs_download)
    return gcs_downloads

  def _get_git_repos(self):
    """Return list of repos to checkout."""
    git_repos_str = utils.get_env_var('ROGUE_GIT_REPOS')
    git_repos = []
    for repo_entry in git_repos_str.split(','):
      repo_parts = repo_entry.split(';')
      git_repo = {}
      if len(repo_parts) >= 2:
        git_repo['local_path'] = repo_parts[0]
        git_repo['url'] = repo_parts[1]
      if len(repo_parts) >= 3:
        git_repo['branch'] = repo_parts[2]
      if len(repo_parts) >= 4:
        git_repo['sha_hash'] = repo_parts[3]
      git_repos.append(git_repo)
    return git_repos


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--gce_nvme_raid',
      type=str,
      default=None,
      help='If set create raid 0 array with devices at disk_dir')
  parser.add_argument(
      '--docker_file',
      type=str,
      default='docker/Dockerfile',
      help='Path to docker build file.')
  parser.add_argument(
      '--data_dir', type=str, default='/data', help='Directory to store data.')

  FLAGS, unparsed = parser.parse_known_args()

  setup_runner = SetupRunner(
      FLAGS.docker_file,
      gce_nvme_raid=FLAGS.gce_nvme_raid,
      data_dir=FLAGS.data_dir)
  setup_runner.setup()
