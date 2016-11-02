# How to run this benchmark  

1. Set up a Kubernetes cluster, either on [Google Compute Engine](http://kubernetes.io/docs/getting-started-guides/gce/)
or on Google Container Engine.

2. Build the docker image which copies the python benchmark or simply use the already-built image 
`gcr.io/tensorflow/alexnet_inference_distributed`.

3. Edit the template file `template.yaml.jinja` with your docker image name, csv output and GCS credentials.

4. You need the [`render_template.py`](https://github.com/tensorflow/ecosystem/blob/master/render_template.py)
script to expand the template to a real Kubernetes config file.
Copy the script somewhere and run:  

  ```sh
  python render_template.py template.yaml.jinja | kubectl create -f -
  ```
  The command also brings up the Tensorflow cluster.
  
5. You can inspect the worker's stdout and stderr by first finding output its pod:

  ```sh
  kubectl get pods
  ```
  Its output looks like:  
  ```
  NAME                     READY     STATUS    RESTARTS   AGE
  alexnet-ps-0-v5fbl       1/1       Running   0          39s
  alexnet-ps-1-kpaup       1/1       Running   0          39s
  alexnet-worker-0-t72v7   1/1       Running   0          39s
  ```  
  
  Then run the following command to inspect the worker's output:
  ```sh
  kubectl logs alexnet-worker-0-t72v7
  ```

6. Shutdown the cluster:  

  ```sh
  python render_template.py template.yaml.jinja | kubectl delete -f -
  ```
