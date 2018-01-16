FROM tensorflow/tensorflow:nightly-gpu

RUN apt-get update && apt-get install -y python-pip && pip install google-cloud
COPY tf_cnn_benchmarks/ ./tf_cnn_benchmarks/
RUN touch tf_cnn_benchmarks/__init__.py
RUN mkdir ./util/
COPY util/ ./util/
ENTRYPOINT ["python", "-m", "tf_cnn_benchmarks.tf_cnn_benchmarks"]
