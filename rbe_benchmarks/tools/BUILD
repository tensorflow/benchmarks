package(default_visibility = ["//visibility:public"])

load("@rbe-it//skylark:integration_tests.bzl", "external_sut_component", "sut_component", "integration_test")
load("@rbe-it//skylark:toolchains.bzl", "toolchain_container_images")

sh_binary(
    name = "benchmark_script",
    srcs = ["benchmark.sh"],
    data = [
        ":benchmark_cpu_config_base.yml",
        ":generate_yml.py",
        ":k8s_tensorflow_lib.py",
        ":kubectl_util.py",
        ":run_distributed_benchmarks.py",
    ],
)

external_sut_component(
    name = "cpu_cluster",
    docker_image = toolchain_container_images()["tensorflow"],
    prepares = [{
        "program": ":cpu_prepare.sh",
        "args": [
            # gcloud cluster prefix
            "cpu-benchmark-cluster",
        ],
        "output_properties": [
            "cluster",
        ],
    }],
    setups = [{
        "program": ":setup.sh",
        "args": [
            # gcloud cluster
            "{prep#cluster}",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
        ],
        "output_properties": ["cluster_name"],
    }],
    teardowns = [{
        "program": ":teardown.sh",
        "args": [
            # gcloud cluster
            "{cluster}",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
        ],
    }],
)

integration_test(
    name = "cpu_benchmark",
    suts = {
        ":cpu_cluster": "k8s",
    },
    test = {
        "program": ":benchmark_script",
        "args": [
            # gcloud cluster prefix
            "{k8s#cluster_name}",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
            # docker image
            "gcr.io/tensorflow-testing/benchmarks/tf-cnn-benchmark-resnet50:test",
            # benchmark config file
            "benchmark_cpu_config_base.yml",
            # benchmark config output
            "/tmp/generated_cpu_benchmark.yml",
            # benchmark name
            "tf-cnn-benchmark-trivial",
        ],
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_timeout = 1200,
    test_type = "MultiMachine",
)

integration_test(
    name = "cpu_benchmark_with_cluster",
    test = {
        "program": ":benchmark_script",
        "args": [
            # gcloud cluster
            "cpu-benchmark-permanent-cluster",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
            # docker image
            "gcr.io/tensorflow-testing/benchmarks/tf-cnn-benchmark-resnet50:test",
            # benchmark config file
            "benchmark_cpu_config_base.yml",
            # benchmark config output
            "/tmp/generated_cpu_benchmark.yml",
            # benchmark name
            "tf-cnn-benchmark-trivial",
        ],
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_timeout = 300,
)

sh_binary(
    name = "token_generate",
    srcs = ["token_generate.sh"],
    data = [
        ":token_vm_generate.sh",
    ],
)

sh_binary(
    name = "token_print_script",
    srcs = ["token_print.sh"],
)

external_sut_component(
    name = "token_vm",
    docker_image = toolchain_container_images()["tensorflow"],
    prepares = [{
        "program": ":token_vm_prepare.sh",
        "output_properties": [
            "vm_name",
            "file_name",
        ],
    }],
    setups = [{
        "program": ":token_generate",
        "args": [
            "{prep#vm_name}",
            "{prep#file_name}",
            # GCS bucket
            "rbe-tensorflow-benchmarks",
        ],
        "output_properties": ["token"],
    }],
    teardowns = [{
        "program": ":token_vm_teardown.sh",
        "args": [
            "{vm_name}",
            "{file_name}",
            # GCS bucket
            "rbe-tensorflow-benchmarks",
        ],
    }],
)

integration_test(
    name = "token_print",
    suts = {
        ":token_vm": "vm",
    },
    test = {
        "program": ":token_print_script",
        "args": [
            "{vm#token}",
        ],
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_timeout = 600,
    test_type = "MultiMachine",
)

sh_binary(
    name = "gpu_benchmark_script",
    srcs = ["gpu_benchmark.sh"],
    data = [
        ":benchmark_gpu_config_base.yml",
        ":generate_yml.py",
        ":k8s_tensorflow_lib.py",
        ":kubectl_util.py",
        ":run_distributed_benchmarks.py",
    ],
)

integration_test(
    name = "gpu_benchmark_with_cluster",
    test = {
        "program": ":gpu_benchmark_script",
        "args": [
            # gcloud cluster
            "rbe-gpu-benchmark-cluster",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
            # docker image
            "gcr.io/tensorflow-testing/benchmarks/tf-cnn-benchmark-resnet50:test",
            # benchmark config file
            "benchmark_gpu_config_base.yml",
            # cuda lib dir
            "/usr/lib/x86_64-linux-gnu/",
            # nvidia lib dir
            "/usr/lib/nvidia-384",
            # benchmark name
            "tf-cnn-benchmark-resnet50",
            # config location
            "gs://rbe-tensorflow-benchmarks/rbe-gpu-benchmark-cluster.conf",
        ],
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_timeout = 1000,
)

sh_binary(
    name = "create_gpu_cluster",
    srcs = ["create_gpu_cluster.sh"],
    data = [
        ":create_gpu_master.sh",
        ":create_gpu_workers.sh",
        ":setup_gpu_master.sh",
        ":setup_gpu_worker.sh",
    ],
)

external_sut_component(
    name = "gpu_cluster",
    docker_image = toolchain_container_images()["tensorflow"],
    prepares = [{
        "program": ":gpu_prepare.sh",
        "args": [
            # gcloud cluster prefix
            "gpu-test-benchmark-cluster",
            # token gcs bucket,
            "rbe-tensorflow-benchmarks",
            # config gcs bucket,
            "rbe-tensorflow-benchmarks",
        ],
        "output_properties": [
            "cluster",
            "config_location",
            "token",
        ],
    }],
    setups = [{
        "program": ":create_gpu_cluster",
        "args": [
            "-m",
            "{prep#cluster}",
            "-w",
            "{prep#cluster}-worker-",
            "-t",
            "{prep#token}",
            "-f",
            "{prep#config_location}",
        ],
        "output_properties": [
            "cluster_name",
            "config_location",
        ],
    }],
    teardowns = [{
        "program": ":gpu_teardown.sh",
        "args": [
            # gcloud cluster master
            "{cluster}",
            # gcloud cluster worker
            "{cluster}-worker-",
            # config location
            "{config_location}",
        ],
    }],
)

integration_test(
    name = "gpu_benchmark",
    suts = {
        ":gpu_cluster": "cluster",
    },
    test = {
        "program": ":gpu_benchmark_script",
        "args": [
            # gcloud cluster
            "{cluster#cluster_name}",
            # gcloud zone
            "europe-west1-b",
            # gcloud project
            "tensorflow-testing",
            # docker image
            "gcr.io/tensorflow-testing/benchmarks/tf-cnn-benchmark-resnet50:test",
            # benchmark config file
            "benchmark_gpu_config_base.yml",
            # cuda lib dir
            "/usr/lib/x86_64-linux-gnu/",
            # nvidia lib dir
            "/usr/lib/nvidia-384",
            # benchmark name
            "tf-cnn-benchmark-resnet50",
            # config location
            "{cluster#config_location}",
        ],
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_timeout = 3000,
    test_type = "MultiMachine",
)
