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
