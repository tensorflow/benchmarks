load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

py_test(
    name = "sample_benchmark",
    srcs = ["sample_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_logged_benchmark(
    name = "sample_logged_benchmark",
    target = "//tensorflow/tools/test:sample_benchmark",
)
