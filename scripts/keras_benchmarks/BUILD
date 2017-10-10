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
    target = "//benchmarks/scripts/keras_benchmarks:sample_benchmark",
)

py_test(
    name = "mnist_benchmark",
    srcs = ["mnist_mlp.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_logged_benchmark(
    name = "mnist_logged_benchmark",
    target = "//benchmarks/scripts/keras_benchmarks:mnist_benchmark",
)
