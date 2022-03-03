import plotly.graph_objects as go
import plotly.express as px

from trame import state
from trame.layouts import SinglePage
from trame.html import vuetify, plotly

import os
from itertools import chain, combinations

import numpy as np

from UncertainSCI.distributions import BetaDistribution, ExponentialDistribution
from UncertainSCI.model_examples import laplace_ode_1d, sine_modulation
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion


def build_pce():

    ## demo

    Nparams = 3

    p1 = BetaDistribution(alpha=0.5, beta=1.)
    p2 = BetaDistribution(alpha=1., beta=0.5)
    p3 = BetaDistribution(alpha=1., beta=1.)

    plabels = ['a', 'b', 'z']

    # # Polynomial order
    order = 5

    N = 100
    x, model = laplace_ode_1d(Nparams, N=N)


    pce = PolynomialChaosExpansion(distribution=[p1, p2, p3], order=order, plabels=plabels)
    pce.generate_samples()

    print('This queries the model {0:d} times'.format(pce.samples.shape[0]))

    model_output = np.zeros([pce.samples.shape[0], N])
    for ind in range(pce.samples.shape[0]):
        model_output[ind, :] = model(pce.samples[ind, :])
    pce.build(model_output=model_output)

    ## Postprocess PCE: statistics are computable:
#    mean = pce.mean()
#    stdev = pce.stdev()
#
#    global_sensitivity, variable_interactions = pce.global_sensitivity()
#    quantiles = pce.quantile([0.25, 0.5, 0.75]) #  0.25, median, 0.75 quantile

    fig = plt.figure(**figure_size())
    fig.subplots_adjust(hspace=1)
    
    ax1 = fig.add_subplot(4, 1, 1)
    ax1 = mean_stdev_plot(pce, ensemble=50,ax =ax1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2= quantile_plot(pce,ax=ax2, xvals = x, xlabel='$x$')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3 = piechart_sensitivity(pce, ax = ax3)
    
    return fig


def contour_plot():
    """https://plotly.com/python/contour-plots/"""
    return go.Figure(
        data=go.Contour(
            z=[
                [10, 10.625, 12.5, 15.625, 20],
                [5.625, 6.25, 8.125, 11.25, 15.625],
                [2.5, 3.125, 5.0, 8.125, 12.5],
                [0.625, 1.25, 3.125, 6.25, 10.625],
                [0, 0.625, 2.5, 5.625, 10],
            ]
        )
    )


def bar_plot(color="Gold"):
    return go.Figure(data=go.Bar(y=[2, 3, 1], marker_color=color))


def scatter():
    df = px.data.iris()

    fig = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        title="Using The add_trace() method With A Plotly Express Figure",
    )

    fig.add_trace(
        go.Scatter(
            x=[2, 4],
            y=[4, 8],
            mode="lines",
            line=go.scatter.Line(color="gray"),
            showlegend=False,
        )
    )

    return fig


PLOTS = {
    "Contour": contour_plot,
    "Bar": bar_plot,
    "Scatter": scatter,
}

def on_event(type, e):
    print(type, e)


html_plot = None
layout = SinglePage("Plotly")
layout.title.set_text("trame ❤️ plotly")

with layout.toolbar:
    vuetify.VSpacer()
    vuetify.VSelect(
        v_model=("active_plot", "Contour"),
        items=("plots", list(PLOTS.keys())),
        hide_details=True,
        dense=True,
    )

with layout.content:
    with vuetify.VContainer(fluid=True):
        with vuetify.VRow(dense=True):
            vuetify.VSpacer()
            html_plot = plotly.Plotly(
                "demo",
                display_mode_bar=("true",),
                selected=(on_event, "['selected', VuePlotly.safe($event)]"),
                # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                # selecting=(on_event, "['selecting', $event]"),
                # unhover=(on_event, "['unhover', $event]"),
            )
            vuetify.VSpacer()


@state.change("active_plot")
def update_plot(active_plot, **kwargs):
    html_plot.update(PLOTS[active_plot]())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    update_plot("Contour")
    layout.start()
