/**
 * @fileoverview Provides a way to create a mean latency chart based on a
 * csv file with latency data.
 */

/**
 * Constructor.
 * @param {string} svg_element_id svg element to add the chart to.
 * @param {string} latency_csv_file File to read input data from. The file
 *     must have lines in the following format:
 *     (Forward|Forward-Backward),timestamp,num_batches,mean,sd
 */
var CsvLatencyChart = function(svg_element_id, latency_csv_file) {
  this.svg_element_id = svg_element_id;
  this.latency_csv_file = latency_csv_file;
};

/**
 * Adds data to the given plots.
 */
CsvLatencyChart.prototype.addData_ = function(
    plotForward, plotForwardBackward) {
  d3.text(this.latency_csv_file, function(data) {
    data = d3.csv.parseRows(data);
    const parseDate = d3.time.format('%Y-%m-%d %H:%M:%S').parse;
    let forwardBenchmarks = [];
    let forwardBackwardBenchmarks = [];
    for (var i = 0; i < data.length; i++) {
      const name = data[i][0];
      const timestamp = data[i][1];
      const mean_latency = data[i][3];
      // Timestamp has the format: 2016-08-31 23:38:55.159320
      // However, we can't parse this date format using d3 time
      // functions, so we remove everything after the dot before parsing.
      const dateUpToSeconds = timestamp.split('.')[0]
      if (name == 'Forward') {
        forwardBenchmarks.push(
            {name: name, timestamp: parseDate(dateUpToSeconds),
             mean_latency: +mean_latency});
      } else {
        forwardBackwardBenchmarks.push(
            {name: name, timestamp: parseDate(dateUpToSeconds),
             mean_latency: +mean_latency});
      }
    }
    plotForward.addDataset(
        new Plottable.Dataset(forwardBenchmarks, {name: 'Forward'}));
    plotForwardBackward.addDataset(
        new Plottable.Dataset(
            forwardBackwardBenchmarks, {name: 'Forward-Backward'}));
  });
};

/**
 * Create the chart.
 */
CsvLatencyChart.prototype.makeChart = function() {
  const xScale = new Plottable.Scales.Time();
  const yScaleForward = new Plottable.Scales.Linear();
  const yScaleForwardBackward = new Plottable.Scales.Linear();

  const plotForward = new LatencyChart(
      'Forward pass per-batch latency', 'Mean latency (sec)',
      xScale, yScaleForward);
  const plotForwardBackward = new LatencyChart(
      'Forward-backward pass per-batch latency', 'Mean latency (sec)',
      xScale, yScaleForwardBackward);

  this.addData_(plotForward, plotForwardBackward);

  const table = new Plottable.Components.Table([
      [plotForward.table],
      [plotForwardBackward.table]
  ]);
  table.renderTo(this.svg_element_id);

  plotForward.addTooltip();
  plotForwardBackward.addTooltip();
  new Plottable.Interactions.Click()
      .attachTo(plotForward.linePlot)
      .onClick(function(p) {
         plotForward.updateForPosition(p);
         plotForwardBackward.updateForPosition(p);
      });
  new Plottable.Interactions.Click()
      .attachTo(plotForwardBackward.linePlot)
      .onClick(function(p) {
         plotForward.updateForPosition(p);
         plotForwardBackward.updateForPosition(p);
      });
};
