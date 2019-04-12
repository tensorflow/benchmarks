
import json
import os
import sys
from absl import app
from pprint import pprint

def init_string(string):
    string = string.replace(" ","")
    string = string.replace("[","")
    string = string.replace("]","")
    string = string.replace("\"","")
    return string

def get_tf_config():
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    return tf_config

def get_cluster_spec(tf_config):
    cluster_spec = tf_config.get("cluster", {})
    return cluster_spec

def get_task_env(tf_config): 
    task_env = tf_config.get("task", {})
    return task_env

def get_worker():
    tf_config = get_tf_config()
    if tf_config:
        cluster_spec = get_cluster_spec(tf_config)
        if cluster_spec:
            workers = init_string(json.dumps(cluster_spec.get("worker", '')))
            return workers
    return NULL

def get_ps():
    tf_config = get_tf_config()
    if tf_config:
        cluster_spec = get_cluster_spec(tf_config)
        if cluster_spec:
            pses = init_string(json.dumps(cluster_spec.get("ps", '')))
            return pses
    return NULL

def get_task_type():
    tf_config = get_tf_config()
    if tf_config:
        task_env = get_task_env(tf_config)
        if task_env:
            task_type = init_string(json.dumps(task_env.get("type", '')))
            return task_type
    return NULL

def get_task_id():
    tf_config = get_tf_config()
    if tf_config:
        task_env = get_task_env(tf_config)
        if task_env:
            task_id = int(task_env.get("index", 0))
            return task_id
    return NULL

def get_rpc_layer():
    tf_config = get_tf_config()
    if tf_config:
        rpc_layer = tf_config.get("rpc_layer", 'grpc')
        return rpc_layer
    return NULL


def dump_tf_config():
    tf_config = get_tf_config()
    pprint(tf_config)


dump_tf_config()
pprint(get_worker())
pprint(get_ps())
pprint(get_task_type())
pprint(get_task_id())
pprint(get_rpc_layer())


def main(_):
 other_opt = ' '

 for i in range(2, len(sys.argv)):
    other_opt += '--' + sys.argv[i] + ' ' 

 pprint(other_opt)

 if sys.argv[2] == 'help':
    os.system("python %s/tf_cnn_benchmarks.py --help" % ( sys.argv[1]))
 elif sys.argv[2] == 'variable_update=parameter_server':
    os.system("python %s/tf_cnn_benchmarks.py \
            --worker_hosts=%s \
            --ps_hosts=%s \
            --job_name=%s \
            --task_index=%d \
            --server_protocol=%s \
            %s "\
            % (   sys.argv[1], \
                  get_worker(), \
                  get_ps(), \
                  get_task_type(), \
                  get_task_id(), \
                  get_rpc_layer(), \
                  other_opt \
             ))
 else:
    task_type = get_task_type()
    if task_type == 'ps':
        task_type = 'controller'
    os.system("python %s/tf_cnn_benchmarks.py \
            --worker_hosts=%s \
            --controller_host=%s \
            --job_name=%s \
            --task_index=%d \
            --server_protocol=%s \
            %s "\
            % (   sys.argv[1], \
                  get_worker(), \
                  get_ps(), \
                  task_type, \
                  get_task_id(), \
                  get_rpc_layer(), \
                  other_opt \
             ))


if __name__ == '__main__':
    app.run(main)
