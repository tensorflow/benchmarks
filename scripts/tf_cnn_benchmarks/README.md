# tf_cnn_benchmarks: High performance benchmarks

tf_cnn_benchmarks contains implementations of several popular convolutional
models, and is designed to be as fast as possible. tf_cnn_benchmarks supports
both running on a single machine or running in distributed mode across multiple
hosts. See the [High-Performance models
guide](https://www.tensorflow.org/performance/performance_models) for more
information.

These models utilize many of the strategies in the [TensorFlow Performance
Guide](https://www.tensorflow.org/performance/performance_guide). Benchmark
results can be found [here](https://www.tensorflow.org/performance/benchmarks).

These models are designed for performance. For models that have clean and
easy-to-read implementations, see the [TensorFlow Official
Models](https://github.com/tensorflow/models/tree/master/official).

## Getting Started

To run ResNet50 with synthetic data without distortions with a single GPU, run

```
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server
```

Note that tf_cnn_benchmarks currently requires the latest nightly version of
TensorFlow. You can install the nightly version by running
`pip install tf-nightly-gpu` in a clean environment, or by installing TensorFlow
from source.

Some important flags are

*   model: Model to use, e.g. resnet50, inception3, vgg16, and alexnet.
*   num_gpus: Number of GPUs to use.
*   data_dir: Path to data to process. If not set, synthetic data is used. To
    use Imagenet data use these
    [instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started)
    as a starting point.
*   batch_size: Batch size for each GPU.
*   variable_update: The method for managing variables: parameter_server
    ,replicated, distributed_replicated, independent
*   local_parameter_device: Device to use as parameter server: cpu or gpu.

To see the full list of flags, run `python tf_cnn_benchmarks.py --help`.
