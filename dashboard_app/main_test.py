# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import main
import unittest
import urllib

class TestMain(unittest.TestCase):

  def testArgumentInvalidFormat(self):
    self.assertEqual('', main.argument_name(''))
    self.assertEqual('', main.argument_name('arg=val'))
    self.assertEqual('', main.argument_name('-arg=val'))
    self.assertEqual('', main.argument_name('--argval'))
    self.assertEqual('', main.argument_name('--=val'))
    self.assertEqual('', main.argument_name('--='))

  def testArgumentValidFormat(self):
    self.assertEqual('abc', main.argument_name('--abc=123'))
    self.assertEqual('a', main.argument_name('--a=123'))

  def testIndexPage(self):
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/')
    self.assertEqual(200, r.status_code)
    self.assertIn('sample_logged_benchmark', r.data.decode('utf-8'))

  def testTestPage_InvalidTest(self):
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/test/abc')
    self.assertEqual(200, r.status_code)
    self.assertIn('No data for benchmark', str(r.data))

  def testTestPage_SampleTest(self):
    main.app.testing = True
    client = main.app.test_client()
    sample_benchmark_name = '//tensorflow/examples/benchmark:sample_logged_benchmark'

    r = client.get(
        '/test/%252F%252Ftensorflow%252Fexamples%252Fbenchmark%253Asample_logged_benchmark')
    self.assertEqual(200, r.status_code)
    self.assertIn(
        'Performance plots for %s' % sample_benchmark_name, str(r.data))

  def testFetchBenchmarkData_InvalidTest(self):
    main.app.testing = True
    client = main.app.test_client()

    r = client.get('/benchmark_data/?test=abc&entry=cde')
    self.assertEqual(200, r.status_code)
    self.assertEqual(b'[]', r.data)

  def testFetchBenchmarkData_SampleTest(self):
    main.app.testing = True
    client = main.app.test_client()

    encoded_benchmark_name = (
        '/test/%252F%252Ftensorflow%252Fexamples%252Fbenchmark%253Asample_logged_benchmark')
    r = client.get('/benchmark_data/?test=%s&entry=SampleBenchmark.sum_wall_time' %
                   encoded_benchmark_name)
    self.assertEqual(200, r.status_code)
    self.assertEqual(b'[]', r.data)


if __name__ == '__main__':
  unittest.main()
