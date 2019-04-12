# TensorFlow benchmarks
This repository contains various TensorFlow benchmarks. Currently, it consists of two projects:

1. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks): The TensorFlow CNN benchmarks contain benchmarks for several convolutional neural networks.
2. [scripts/keras_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/keras_benchmarks): The Keras benchmarks contain benchmarks for several models using Keras. Note this project is deprecated and unmaintained.

---

### Test Instruction

This is a brief summary for benchmarks testing.

##### Test on bare machine

0. Set python path

   ~~~shell
   export PYTHONPATH=/home/ubuntu/zcg/models
   ~~~

1. Start controller

   For distributed all reduce strategies, you have to start a controller for the workers. The controller take charge of the tensorflow graph construction and results display.

   ~~~shell
   python ~/zcg/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
           --worker_hosts=${WORKER1},... \
           --controller_host=${CONTROLLER_HOST} \
           --job_name=controller \
           --variable_update=${VARIABLE_UPDATE} \
           --local_parameter_device=cpu \
           --use_fp16 --batch_size=${BATCH_SIZE_PER_GPU} \
           --force_gpu_compatible \
           --num_gpus=4 \
           --model=${TRAIN_MODEL} \
           --task_index=0 \
           --server_protocol=${PROTOCOL} \
           --all_reduce_spec=${ALL_REDUCE_ALG} \
           --benchmark_log_dir=/tmp/bench_log
   ~~~

2. Start parameter server

   + Set CUDA_VISIBLE_DEVICES, so that the parameter server will be deploy on cpu.

   ~~~shell
   export CUDA_VISIBLE_DEVICES=
   ~~~

   + Start servers

   ~~~shell
   python ~/zcg/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
       --ps_hosts=${PS1},... \
       --worker_hosts=${WORKER1},... \
       --job_name=ps \
       --variable_update=${VARIABLE_UPDATE} \
       --local_parameter_device=cpu \
       --use_fp16 --batch_size=${BATCH_SIZE_PER_GPU} \
       --force_gpu_compatible \
       --num_gpus=4 \
       --model=${TRAIN_MODEL} \
       --task_index=0 \
       --server_protocol=${PROTOCOL} \
       --benchmark_log_dir=/tmp/bench_log
   ~~~

3. Start worker

   + Set CUDA_VISIBLE_DEVICES, so that the workers will be deploy on gpu.

   ~~~shell
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ~~~

   + start worker

   ~~~shell
   python ~/zcg/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
           --ps_hosts=${PS1},${PS2} \ # be used when using ps strategy
           --worker_hosts=${WORKER1},${WORKER2} \
           --controller_host=${CONTROLLER_HOST} \ # be used when using distributed all reduce
           --job_name=worker \
           --variable_update=${VARIABLE_UPDATE} \
           --local_parameter_device=cpu \
           --use_fp16 --batch_size=${BATCH_SIZE_PER_GPU} \
           --force_gpu_compatible \
           --num_gpus=4 \
           --model=${TRAIN_MODEL} \
           --task_index=0 \
           --server_protocol=${PROTOCOL} \
           --all_reduce_spec=${ALL_REDUCE_ALG}\
           --benchmark_log_dir=/tmp/bench_log
           --allreduce_merge_scope=64 # be used when using collective all reduce
   ~~~

##### Test in container

0. Set tensorflow image.

   ~~~shell
   GPU_TF_IMAGE_PY2=172.16.0.150:5000/clustar-lalalapotter/tensorflow:PY2-19-04-11-03-47-39
   ~~~

1. Use submit-tf command to submit tensorflow tasks, e.g:

   ~~~shell
   submit-tf --name=${TIME} \
   	--gpus=${GPU_NUM} \
   	--ps=${PS_NUM} \
   	--psMemory=64Gi \
   	--workers=${WK_NUM} \
   	--workerMemory=64Gi \
   	--rpcLayer=${RPC_LAYER} \
       "python ${CODE_DIR}/runme.py \
           ${CODE_DIR} \
           ${CMD} "
   # CMD EXAMPLE:
   # " variable_update=parameter_server \
   #         model=${MODEL} \
   #         device=${DEVICE} \
   #         num_gpus=${GPU_NUM} \
   #         batch_size=${BATCHSIZE} \
   #         use_fp16 \
   #         force_gpu_compatible \
   #         local_parameter_device=cpu "
   ~~~

   or you can just use runbenchmarks.sh, e.g:

   ~~~shell
   ./runbenchmarks.sh ps 2 2 grpc+gdr vgg16 256
   ~~~

2. Check running status.

   0. Check the status of pods

      ~~~shell
      kubectl get pods -o wide
      ~~~

   1. Check the logs of pods

      ~~~shell
      kubectl logs pods_name (-f)
      ~~~

   2. Delete completed pods

      ~~~shell
      kubectl delete pods_name
      ~~~

   3. Delete running or pending pods

      ~~~shell
      arena list
      arena delete arena_name
      ~~~

   4. Other useful command

      ~~~shell
      # start bash in a pod
      kubectl exec -it pods_name bash
      # show the description of a pod
      kubectl describe pods_name
      ~~~