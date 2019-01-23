# PerfZero

This libary provides solutions to benchmark Tensorflow performance


# Getting Started

## Setup benchmark environment

The commands below builds the docker image named `temp/tf-gpu` which contains the
libraries needed for benchmark. It also downloads data and checkouts dependent libraries.

This setup script only needs to be executed when we need to create docker image for the
first time, or when we need to update docker image based on the latest
Tensorflow nighly build.


```
source staging/scripts/setup_env.sh

python3 staging/lib/setup.py

```



## Run benchmark

The commands below run benchmarks specificed in `staging/scripts/setup_env.sh`


```
nvidia-docker run -it -v $(pwd):/workspace \
-v /data:/data \
-e ROGUE_ZERO_PLATFORM_TYPE \
-e ROGUE_CODE_DIR \
-e ROGUE_TEST_CLASS \
-e ROGUE_TEST_METHODS \
-e ROGUE_PYTHON_PATH \
-e ROGUE_REPORT_PROJECT \
-e ROGUE_TEST_ENV \
-e ROGUE_PLATFORM \
-e ROGUE_PLATFORM_TYPE \
-e ROGUE_GIT_REPOS \
temp/tf-gpu \
python /workspace/staging/lib/benchmark.py

```

## Run all unit tests

```
cd staging/lib

python -B -m unittest discover -p "*_test.py"
```

## Format python code style

```
find lib -name *.py -exec pyformat --in_place {} \;
```

