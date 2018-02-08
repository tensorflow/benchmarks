FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update && apt-get install -y python3-pip && pip3 install google-cloud && pip3 install absl-py
COPY tf_cnn_benchmarks/ ./
RUN touch __init__.py
RUN mkdir ./util/
COPY util/ ./util/
ENTRYPOINT ["python3", "-m", "tf_cnn_benchmarks"]
