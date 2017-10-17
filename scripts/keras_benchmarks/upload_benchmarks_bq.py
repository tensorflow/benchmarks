from google.cloud import bigquery
from google.cloud import storage
import uuid
import urllib3

urllib3.disable_warnings()


def upload_metrics_to_bq(test_name, total_time, num_iters, sample_type=None):
  bigquery_client = bigquery.Client()
  dataset = bigquery_client.dataset('tf_test_dataset')
  table = dataset.table('TestBqTable')

  table.reload()

  query = """\
  INSERT tf_test_dataset.Benchmarks (test_name,recorded_time,metrics_data) VALUES(@testname,CURRENT_TIMESTAMP(),(@totaltime,@numiters,@sampletype))
  """

  query_job = bigquery_client.run_async_query(
      str(uuid.uuid4()),
      query,
      query_parameters=(
        bigquery.ScalarQueryParameter('testname', 'STRING', test_name),
        bigquery.ScalarQueryParameter('totaltime', 'TIME', total_time),
        bigquery.ScalarQueryParameter('numiters', 'INTEGER', num_iters),
        bigquery.ScalarQueryParameter('sampletype', 'STRING', sample_type)))
  query_job.use_legacy_sql = False

  query_job.begin()
  query_job.result()  # Wait for job to complete.



