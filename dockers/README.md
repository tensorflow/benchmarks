## Build

The following command builds a custom image for running tensorflow-benchmarks and Horovod.

```bash
$ docker build -f Dockerfile_hvd_tf_benchmarks -t hvd-bench:latest .
```

## Train with synthetic data

The following command run tensorflow-benchmarks for ResNet50 using 4 local GPUs and synchronise through Horovod. Check more about [Horovod Docker](https://github.com/horovod/horovod/blob/master/docs/docker.md).

```bash
$ nvidia-docker run --runtime=nvidia -it hvd-bench:latest
root@243d81c298a9:/benchmarks/scripts/tf_cnn_benchmarks# mpirun -np 4 -H localhost:4 python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=horovod
```

If you don't run your container in privileged mode, you may see the following message:

```bash
[a8c9914754d2:00040] Read -1, expected 131072, errno = 1
```

You can ignore this message, or filter out this message using `|& grep -v "Read -1"` as follows:

```bash
$ mpirun -np 4 -H localhost:4 python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=horovod |& grep -v "Read -1"
```

## Train with real ImageNet

You often need to mount the ImageNet dataset into the container. Assuming
the host's dataset is at: `/data/tf/imagenet/records`, and you want to
mount it as a directory within the container at `/data`. We can mount this dataset
using the `-v` option as follows:

```bash
docker run --runtime=nvidia -v /data/tf/imagenet/records:/data -it hvd-bench:latest 
```

We can train the `tf_cnn_benchmark` using the real ImageNet dataset as follows:

```bash
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server --data_name=imagenet --data_dir=/data/
```

Here we use the `--data_name` option to specify the use of imagenet,
and the `--data_dir` option to specify the path to the dataset.