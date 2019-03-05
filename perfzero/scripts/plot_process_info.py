#!/usr/bin/python
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Plot graph showing process metric values over time"""

from __future__ import print_function

import argparse
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.ticker as tick

colors=['b', 'r', 'g', 'c', 'pink']

def visualize(file_path):

  entries = []
  with open(file_path) as f:
    entries = [json.loads(line) for line in f.readlines() if line.strip()]

  if not entries:
    print('There is no data in file {}'.format(file_path))
    return

  pdf = backend_pdf.PdfPages("process_info.pdf")
  idx = 0
  names = [name for name in entries[0].keys() if name != 'time']
  times = [entry['time'] for entry in entries]

  for name in names:
    values = [entry[name] for entry in entries]
    fig = plt.figure()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(tick.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,3))
    plt.plot(times, values, colors[idx % len(colors)], marker='x', label=name)
    plt.xlabel('Time (sec)')
    plt.ylabel(name)
    plt.ylim(ymin=0)
    plt.legend(loc = 'upper left')
    pdf.savefig(fig)
    idx += 1

  plt.show()
  pdf.close()
  print('Generated process_info.pdf from {}'.format(file_path))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(usage='plot_process_info.py <path_to_file>' )
  parser.add_argument('file_path', type=str)
  flags = parser.parse_args(sys.argv[1:])


  visualize(flags.file_path)



