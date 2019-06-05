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

import logging
import os


def add_setup_parser_arguments(parser):
  """Add arguments to the parser used by the setup.py."""
  parser.add_argument(
      '--dockerfile_path',
      default='docker/Dockerfile_ubuntu_1804_tf_v1',
      type=str,
      help='''Build the docker image using docker file located at the ${pwd}/${dockerfile_path} if
      it exists, where ${pwd} is user's current work directory. Otherwise, build
      the docker image using the docker file located at path_to_perfzero/${dockerfile_path}.
      ''')
  parser.add_argument(
      '--workspace',
      default='workspace',
      type=str,
      help='''The gcloud key file will be downloaded under directory path_to_perfzero/${workspace}
      ''')
  parser.add_argument(
      '--gcloud_key_file_url',
      default='',
      type=str,
      help='''DEPRECATED: Use --gcloud_key_file_url of setup.py instead.
      The gcloud key file url. When specified, it will be downloaded to the
      directory specified by the flag --workspace. Each url can start with file://, gcs://, http:// or https://.
      ''')
  parser.add_argument(
      '--root_data_dir',
      default='/data',
      type=str,
      help='The directory which should contain the dataset required by the becnhmark method.'
      )
  parser.add_argument(
      '--gce_nvme_raid',
      default=None,
      type=str,
      help='If set to non-empty string, create raid 0 array with devices at the directory specified by the flag --root_data_dir'
      )
  parser.add_argument(
      '--tensorflow_pip_spec',
      default=None,
      type=str,
      help='''The tensorflow pip package specfication. The format can be either ${package_name}, or ${package_name}==${package_version}.
      Example values include tf-nightly-gpu, and tensorflow==1.12.0. If it is specified, the corresponding tensorflow pip package/version
      will be installed. Otherwise, the default tensorflow pip package specified in the docker file will be installed.
      ''')


def add_benchmark_parser_arguments(parser):
  """Add arguments to the parser used by the benchmark.py."""
  parser.add_argument(
      '--use_cached_site_packages',
      action='store_true',
      help='If set, skip git pull for dependent git repositories if it already exists in path_to_perfzero/${workspace}/site-packages'
      )
  parser.add_argument(
      '--gcs_downloads',
      default=None,
      type=str,
      help='This flag is deprecated. Use the flag --data_downloads instead')
  parser.add_argument(
      '--git_repos',
      default=None,
      type=str,
      help='''A string representing git repositories to checkout. The format is url_1;branch_1;hash_1,url_2;branch_2;hash_2,...
      Git repositories will be checked-out under directory path_to_perfzero/${workspace}/site-packages,
      where ${workspace} either defaults to 'workspace', or takes the value of the flag --workspace.
      branch and hash can be skipped if user wants to use the head of the master branch,
      which shortens the format to url_1,url_2,...
      ''')
  parser.add_argument(
      '--benchmark_methods',
      action='append',
      default=[],
      type=str,
      help='''This string specifies the benchmark_method to be executed. The flag can be specified multiple times in which case
      the union of methods matched by these flags will be executed. The format can be module_path.class_name.method_name in which
      case the corresponding method is executed. The format can also be module_path.class_name.filter:regex_pattern, in which case all methods
      of the given class whose method name matches the given regular expression are executed.
      ''')
  parser.add_argument(
      '--ml_framework_build_label',
      default=None,
      type=str,
      help='A string that identified the machine learning framework build, e.g. nightly-gpu-build'
      )
  parser.add_argument(
      '--execution_label',
      default=None,
      type=str,
      help='A string that identified the benchmark execution type, e.g. test, prod'
      )
  parser.add_argument(
      '--platform_name',
      default=None,
      type=str,
      help='A string that identified the computing platform, e.g. gcp, aws'
      )
  parser.add_argument(
      '--system_name',
      default=None,
      type=str,
      help='A string that identified the hardware system, e.g. n1-standard-64-8xV100'
      )
  parser.add_argument(
      '--output_gcs_url',
      default=None,
      type=str,
      help='''If specified, log files generated by the benchmark execution will be uploaded to output_gcs_url/${execution_id},
      where ${execution_id} is a string that generated by PerfZero which uniquely identifies the execution of one benchmark method
      ''')
  parser.add_argument(
      '--bigquery_project_name',
      default=None,
      type=str,
      help='''If both --bigquery_project_name and --bigquery_dataset_table_name are specified, for each benchmark method, the benchmark
      summary will be uploaded to the specified bigquery table whose schema is defined in perfzero/scripts/create_big_table.txt.
      The value of each field can in turn be a json-formatted string. See README.md for example output.
      ''')
  parser.add_argument(
      '--bigquery_dataset_table_name',
      default=None,
      type=str,
      help='''If both --bigquery_project_name and --bigquery_dataset_table_name are specified, for each benchmark method, the benchmark
      summary will be uploaded to the specified bigquery table whose schema is defined in perfzero/scripts/create_big_table.txt.
      The value of each field can in turn be a json-formatted string. See README.md for example output.
      ''')
  parser.add_argument(
      '--python_path',
      default=None,
      type=str,
      help='''A string of format path_1,path_2,... For each ${path} specified in the string,
      path_to_perfzero/${workspace}/site-packages/${path} will be added to python path so that libraies downloaded by --git_repos can
      be loaded and executed.
      ''')
  parser.add_argument(
      '--workspace',
      default='workspace',
      type=str,
      help='''The log files, gcloud key file and git repositories will be generated and downloaded under the
      directory path_to_perfzero/${workspace}
      ''')
  parser.add_argument(
      '--root_data_dir',
      default='/data',
      type=str,
      help='The directory which should contain the dataset required by the becnhmark method.'
      )
  parser.add_argument(
      '--data_downloads',
      default=None,
      type=str,
      help='''A string of format url_1;relative_path_1,url_2;relative_path_2,...
      Data will be copied from ${url} to ${root_data_dir}/${relative_path}. ${relative_path} can be skipped if it is the same as the
      base name of the url, which shortens the format to url_1,url_2,... ${root_data_dir} is specified by the flag --root_data_dir.
      File will be de-compressed in ${root_data_dir} if its name ends with 'gz'. Only url prefixed with gcs, http or https are supported.
      Each url can start with file://, gcs://, http:// or https://.
      ''')
  parser.add_argument(
      '--gcloud_key_file_url',
      default='gs://tf-performance/auth_tokens/benchmark_upload_gce.json',
      type=str,
      help='''The gcloud key file url. When specified, it will be downloaded to the
      directory specified by the flag --workspace. Each url can start with file://, gcs://, http:// or https://.
      The key file will then be activated and used as gcloud authentication credential.
      ''')
  parser.add_argument(
      '--debug',
      action='store_true',
      help='If set, use debug level logging. Otherwise, use info level logging'
      )
  parser.add_argument(
      '--profiler_enabled_time',
      default=None,
      type=str,
      help='''A string of format begin_time_1:end_time_1,begin_time_2:end_time_2,.... PerfZero will start to collect profiler
      data ${begin_time} sec after benchmark method execution starts. The data collection continues for ${end_time - begin_time}
      sec or until the benchmark method execution finishes, whichever occurs first. If ${end_time} is not explicitly
      specified, it is assumed to be MAX_LONG.
      ''')
  parser.add_argument(
      '--execution_id',
      default=None,
      type=str,
      help='A string that uniquely identifies the benchmark execution.')


class PerfZeroConfig(object):
  """Creates and contains config for PerfZero."""

  def __init__(self, mode, flags=None):
    self.mode = mode
    self.flags = flags
    if mode == 'flags':
      self.gcs_downloads_str = flags.gcs_downloads
      self.data_downloads_str = flags.data_downloads
      self.git_repos_str = flags.git_repos
      self.benchmark_method_patterns = []
      for value in flags.benchmark_methods:
        self.benchmark_method_patterns.extend(value.split(','))
      self.ml_framework_build_label = flags.ml_framework_build_label
      self.execution_label = flags.execution_label
      self.platform_name = flags.platform_name
      self.system_name = flags.system_name
      self.output_gcs_url = flags.output_gcs_url
      self.bigquery_project_name = flags.bigquery_project_name
      self.bigquery_dataset_table_name = flags.bigquery_dataset_table_name
      self.python_path_str = flags.python_path
      self.workspace = flags.workspace
      self.use_cached_site_packages = flags.use_cached_site_packages
      self.root_data_dir = flags.root_data_dir
      self.gcloud_key_file_url = flags.gcloud_key_file_url
      self.profiler_enabled_time_str = flags.profiler_enabled_time
      self.execution_id = flags.execution_id

      if not flags.benchmark_methods:
        logging.warning('No benchmark method is specified by '
                        '--benchmark_methods')

      if flags.bigquery_project_name and not flags.bigquery_dataset_table_name:
        raise ValueError('--bigquery_project_name is specified but '
                         '--bigquery_dataset_table_name is not')

      if not flags.bigquery_project_name and flags.bigquery_dataset_table_name:
        raise ValueError('--bigquery_dataset_table_name is specified but '
                         '--bigquery_project_name is not')

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
      git_repo['url'] = parts[0]
      # Assume the git url has format */{dir_name}.git
      git_repo['dir_name'] = parts[0].rsplit('/', 1)[-1].rsplit('.', 1)[0]
      git_repo['local_path'] = os.path.join(site_packages_dir,
                                            git_repo['dir_name'])
      if len(parts) >= 2:
        git_repo['branch'] = parts[1]
      if len(parts) >= 3:
        git_repo['git_hash'] = parts[2]
      git_repos.append(git_repo)

    return git_repos


