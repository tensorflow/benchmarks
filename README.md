# Instructions for adding distributed benchmarks to continuous run:

1. You can add your benchmark file under
   [tensorflow/benchmarks/scripts](https://github.com/tensorflow/benchmarks/tree/master/scripts) directory. The benchmark should accept `task_index`, `job_name`, `ps_hosts` and `worker_hosts` flags. You can copy-paste the following flag definitions:

    ```python
    tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
    tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
    tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")
    ```
2. Report benchmark values by calling `store_data_in_json` from your benchmark
   code. This function is defined in
   [benchmark\_util.py](https://github.com/tensorflow/benchmarks/blob/master/scripts/util/benchmark_util.py).
3. Create a Dockerfile that sets up dependencies and runs your benchmark. For
   example, see [Dockerfile.tf\_cnn\_benchmarks](https://github.com/tensorflow/benchmarks/blob/master/scripts/Dockerfile.tf_cnn_benchmarks).
4. Add the benchmark to
   [benchmark\_configs.yml](https://github.com/tensorflow/benchmarks/blob/master/scripts/benchmark_configs.yml)
   * Set `benchmark_name` to a descriptive name for your benchmark and make sure
     it is unique.
   * Set `worker_count` and `ps_count`.
   * Set `docker_file` to the Dockerfile path starting with `benchmarks/`
     directory.
   * Optionally, you can pass flags to your benchmark by adding `args` list.
5. Send PR with the changes to annarev.

Currently running benchmarks:
https://benchmarks-dot-tensorflow-testing.appspot.com/

For any questions, please contact annarev@google.com.
