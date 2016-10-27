/**
 * @fileoverview Combines all components needed to display a line chart for
 * benchmarks.
 * @param {string} title Graph title.
 * @param {string} yLabel Label to use for the y-axis.
 * @param {Plottable.Scale} xScale X-scale for the graph.
 * @param {Plottable.Scale} yScale Y-scale for the graph.
 */
var LatencyChart = function(title, yLabel, xScale, yScale) {
  this.linePlot = new Plottable.Plots.Line()
      .x(function(d) { return d.timestamp; }, xScale)
      .y(function(d) { return d.mean_latency; }, yScale)
      .attr('stroke-width', 3);
  this.pointPlot = new Plottable.Plots.Scatter()
      .x(function(d) { return d.timestamp; }, xScale)
      .y(function(d) { return d.mean_latency; }, yScale)
      .size(10)
      .attr('opacity', 1)
      .addDataset(new Plottable.Dataset());
  this.guideline = new Plottable.Components.GuideLineLayer('vertical')
      .scale(xScale);
  this.xAxis = new Plottable.Axes.Time(xScale, 'bottom')
      .annotationsEnabled(true);
  this.yAxis = new Plottable.Axes.Numeric(yScale, 'left');
  this.title = new Plottable.Components.TitleLabel(title)
      .yAlignment('top').padding(10);
  this.yLabel = new Plottable.Components.AxisLabel(yLabel, '270');
  const group = new Plottable.Components.Group(
      [this.guideline, this.linePlot, this.pointPlot]);
  this.table = new Plottable.Components.Table([
      [null, null, this.title],
      [this.yLabel, this.yAxis, group],
      [null, null, this.xAxis]
  ]);
};

/**
 * Add value tooltip to this chart.
 * Note: this method must be called after the chart has been
 * rendered.
 */
LatencyChart.prototype.addTooltip = function() {
  this.tooltip = new ValueTooltip(this.linePlot);
};

/**
 * Add data to this chart.
 * Data must be an Array with items in the form:
 * {name, timestamp, mean_latency}
 */
LatencyChart.prototype.addDataset = function(data) {
  this.linePlot.addDataset(data);
};

/**
 * Update chart for the given position: display a guideline,
 * x-axis value indicator and show value tooltip at this position.
 * @param {Plottable.Point} position Point to update for.
 */
LatencyChart.prototype.updateForPosition = function(position) {
  const entity = this.linePlot.entityNearest(position);
  if (typeof entity == 'undefined') {
    return;
  }
  const date = new Date(entity.datum.timestamp);
  this.guideline.value(date);
  this.pointPlot.datasets()[0].data([entity.datum]);
  if (typeof this.tooltip != 'undefined') {
    this.tooltip.update(entity.position, entity.datum.mean_latency);
  }
  this.xAxis.annotatedTicks([date]);
};

/**
 * Constructor. Creates a tooltip to display chart value.
 * @param {Plottable.Plots.Line} linePlot Plot to add the tooltip to.
 */
var ValueTooltip = function(linePlot) {
  this.tooltip = d3.select('body').append('div')
      .style('position', 'absolute')
      .style('z-index', '10')
      .style('visibility', 'hidden');
  this.tooltipAnchorSelection = linePlot.foreground().append('circle').attr({
    r: 3,
    opacity: 0
  });
};

/**
 * Display tooltip at the given position.
 * @param {Plottable.Point} position Point to display tooltip at.
 * @param {number} data Numeric data to show.
 */
ValueTooltip.prototype.update = function(position, data) {
  this.tooltipAnchorSelection.attr({
    cx: position.x,
    cy: position.y
  });
  const boundingRect =
      this.tooltipAnchorSelection.node().parentElement.getBoundingClientRect();
  this.tooltip.style(
      'top',
      (boundingRect.top - document.body.getBoundingClientRect().top-10)+'px')
      .style('left',(boundingRect.left+10)+'px');
  this.tooltip.style('visibility', 'visible');
  this.tooltip.text(data.toPrecision(6));
};
