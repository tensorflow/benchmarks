""" Uploads benchmark statistics along with platform used to run the benchmark
to BigQuery."""
from google.cloud import bigquery
import uuid


def upload_metrics_to_bq(test_name, total_time, epochs, batch_size,
    backend_type, backend_version, cpu_num_cores, cpu_memory, cpu_memory_info,
    gpu_count, gpu_platform, platform_type, platform_machine_type,
    keras_version, sample_type=None):
    """ Upload benchmark metrics of a model along with platform specs.

    # Arguments
        test_name: Unique test name for each benchmark.
        total_time: Time taken to run the given number of epochs.
        epochs: Total number of epochs for which the given benchmark was run.
                       We don't count the first epoch since some amount of time is
                       spent creating the graph.
        batch_size: Batch size of samples used in a given epoch.
        backend_type: Backend type used by the Keras models. This is either
                            "tensorflow", "cntk" or "theano".
        backend_version: This is the version "tensorflow", "cntk" or "theano"
                                used by Keras.
        cpu_num_cores: Number of CPU cores of the machine on which the Keras
                              benchmark is run.
        cpu_memory: RAM memory specs of the CPU.
        cpu_memory_info: This is the memory unit of the CPU memory such as
        gpu_count: Number of GPUs used to run the benchmarks.
                                'GB'.
        gpu_platform: The type of GPU used, for e.g "Nvidia Tesla K80"
        platform_type: This is the local or cloud platform used to run the
                              benchmarks.
        platform_machine_type: This can be details about the machine type
                                      for e.g
        keras_version: Version of Keras used to run the benchmark model.
        sample_type: This is a user specified string used to calculate metrics such
                    as "images per epoch" etc.
    """
    bigquery_client = bigquery.Client()
    dataset = bigquery_client.dataset('keras_benchmarks')
    table = dataset.table('benchmarks')
    table.reload()

    query = """\
    INSERT keras_benchmarks.benchmarks (test_id,test_name,recorded_time,\
    metrics,keras_backend,cpu_info,platform_info,keras_version,gpu_info) \
    VALUES(@testid,@testname,CURRENT_TIMESTAMP(),\
    (@metrics_totaltime,@metrics_epochs,@metrics_batch_size,@metrics_sampletype),\
    (@keras_backend_type, @keras_backend_version),\
    (@cpu_info_numcores,@cpu_info_memory, @cpu_info_memory_units),\
    (@platform_info_type,@platform_info_machine_type),\
     @keras_version,\
     (@gpu_info_count,@gpu_info_platform))
    """
    test_id = uuid.uuid4().int >> 80
    query_job = bigquery_client.run_async_query(
        str(uuid.uuid4()),
        query,
        query_parameters=(
          bigquery.ScalarQueryParameter('testid', 'INTEGER', test_id),
          bigquery.ScalarQueryParameter('testname', 'STRING', test_name),
          bigquery.ScalarQueryParameter('metrics_totaltime', 'FLOAT', total_time),
          bigquery.ScalarQueryParameter('metrics_epochs', 'INTEGER', epochs),
          bigquery.ScalarQueryParameter('metrics_batch_size', 'INTEGER', batch_size),
          bigquery.ScalarQueryParameter('metrics_sampletype', 'STRING', sample_type),
          bigquery.ScalarQueryParameter('keras_backend_type', 'STRING', backend_type),
          bigquery.ScalarQueryParameter('keras_backend_version', 'STRING', backend_version),
          bigquery.ScalarQueryParameter('cpu_info_numcores', 'FLOAT', cpu_num_cores),
          bigquery.ScalarQueryParameter('cpu_info_memory', 'FLOAT', cpu_memory),
          bigquery.ScalarQueryParameter('cpu_info_memory_units', 'STRING', cpu_memory_info),
          bigquery.ScalarQueryParameter('platform_info_type', 'STRING', platform_type),
          bigquery.ScalarQueryParameter('platform_info_machine_type', 'STRING', platform_machine_type),
          bigquery.ScalarQueryParameter('keras_version', 'STRING', keras_version),
          bigquery.ScalarQueryParameter('gpu_info_count', 'FLOAT', gpu_count),
          bigquery.ScalarQueryParameter('gpu_info_platform', 'STRING', gpu_platform)))

    query_job.use_legacy_sql = False

    query_job.begin()
    query_job.result()
