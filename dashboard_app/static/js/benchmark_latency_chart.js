// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Provides a way to create a benchmark latency chart.
 */

/**
 * Constructor.
 * @param {string} svg_element_id svg element to add the chart to.
 * @param {string} test_id of the test to plot data for.
 * @param {string} entry_id of the specific test entry to plot.
 */
var BenchmarkLatencyChart = function(svg_element, test_id, entry_id) {
  this.svg_element = svg_element;
  this.test_id = test_id;
  this.entry_id = entry_id;
};

/**
 * Adds data to the given plots.
 */
BenchmarkLatencyChart.prototype.addData_ = function(plot) {
  const encodedTestId = encodeURIComponent(this.test_id);
  const encodedEntryId = encodeURIComponent(this.entry_id);
  const jsonDataUrl =
      '/benchmark_data/?test=' + encodedTestId + '&entry=' + encodedEntryId
  d3.json(jsonDataUrl, function(data) {
    benchmarks = []
    for (var i = 0; i < data.length; i++) {
      const name = this.entry_id;
      const timestamp = new Date(+data[i]['start'] / 1000);
      const mean_latency = data[i]['timing'];
      benchmarks.push(
          {name: name, timestamp: timestamp,
           mean_latency: +mean_latency});
    }
    plot.addDataset(
      new Plottable.Dataset(benchmarks, {name: 'Forward'}));
  });
};

/**
 * Create the chart.
 */
BenchmarkLatencyChart.prototype.makeChart = function() {
  const xScale = new Plottable.Scales.Time();
  const yScaleForward = new Plottable.Scales.Linear();

  const plot = new LatencyChart(
      this.entry_id, 'value',
      xScale, yScaleForward);

  this.addData_(plot);

  const table = new Plottable.Components.Table([[plot.table]]);
  table.renderTo(this.svg_element);

  plot.addTooltip();
  new Plottable.Interactions.Click()
      .attachTo(plot.linePlot)
      .onClick(function(p) {
         plot.updateForPosition(p);
      });
};
