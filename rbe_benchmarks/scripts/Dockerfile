FROM tensorflow/tensorflow:nightly

RUN apt-get update && apt-get install -y python-pip && pip install google-cloud
RUN mkdir ./tf_cnn_benchmarks/
COPY tf_cnn_benchmarks/*.py ./tf_cnn_benchmarks/
RUN touch tf_cnn_benchmarks/__init__.py
COPY run_benchmarks.sh ./run_benchmarks.sh
ENTRYPOINT ["./run_benchmarks.sh"]
