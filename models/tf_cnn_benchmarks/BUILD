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

py_binary(
    name = "tf_cnn_benchmarks",
    srcs = [
        "tf_cnn_benchmarks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cnn_util",
        ":datasets",
        ":model_config",
        ":preprocessing",
        ":variable_mgr",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/nccl:nccl_py",
    ],
)

py_library(
    name = "cnn_util",
    srcs = [
        "cnn_util.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "datasets",
    srcs = [
        "datasets.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_config",
    srcs = [
        "alexnet_model.py",
        "googlenet_model.py",
        "inception_model.py",
        "lenet_model.py",
        "model.py",
        "model_config.py",
        "overfeat_model.py",
        "resnet_model.py",
        "trivial_model.py",
        "vgg_model.py",
    ],
    deps = [
    ],
)

py_library(
    name = "preprocessing",
    srcs = [
        "preprocessing.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "variable_mgr",
    srcs = [
        "variable_mgr.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cuda_py_test(
    name = "tf_cnn_benchmarks_test",
    srcs = ["tf_cnn_benchmarks_test.py"],
    additional_deps = [":tf_cnn_benchmarks"],
    shard_count = 14,
)
