import os

import numpy as np
from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_ode_1d
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

import plotly.graph_objects as go
import plotly.express as px

from trame import state
from trame.layouts import SinglePage
from trame.html import vuetify, plotly



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
mean = pce.mean()
stdev = pce.stdev()

global_sensitivity, variable_interactions = pce.global_sensitivity()
quantiles = pce.quantile([0.25, 0.5, 0.75]) #  0.25, median, 0.75 quantile


def figure_size():
    if state.figure_size is None:
        return {}

    pixel_ratio = state.figure_size.get("pixelRatio")
    dpi = state.figure_size.get("dpi")
    rect = state.figure_size.get("size")
    w_inch = rect.get("width") / dpi / pixel_ratio
    h_inch = rect.get("height") / dpi / pixel_ratio

    return {
        "figsize": (w_inch, h_inch),
        "dpi": dpi,
    }
    

def MeanStd():

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_rev = x[::-1]

    # Line 1
    y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y1_lower = y1_lower[::-1]

    # Line 2
    y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
    y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
    y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
    y2_lower = y2_lower[::-1]

    # Line 3
    y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
    y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
    y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
    y3_lower = y3_lower[::-1]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y1_upper+y1_lower,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y2_upper+y2_lower,
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Premium',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y3_upper+y3_lower,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Ideal',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(0,176,246)',
        name='Premium',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y3,
        line_color='rgb(231,107,243)',
        name='Ideal',
    ))

    fig.update_traces(mode='lines')
    
    return fig


# -----------------------------------------------------------------------------


def Quantiles():

    fig, ax = plt.subplots(**figure_size() )
    
    ax= quantile_plot(pce,ax=ax, xvals = x, xlabel='$x$')


    return fig


# -----------------------------------------------------------------------------


def SensitivityPiechart():

    fig, ax = plt.subplots(**figure_size())
    
    ax = piechart_sensitivity(pce, ax = ax)

    return fig



# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

PLOTS = {
    "MeanStd": MeanStd,
    "Quantiles": Quantiles,
    "Sensitivities": SensitivityPiechart,
}

def on_event(type, e):
    print(type, e)
    


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

html_plot = None
layout = SinglePage("Plotly")
layout.title.set_text("UncertainSCI demo")

with layout.toolbar:
    vuetify.VSpacer()
    vuetify.VSelect(
        v_model=("active_plot", "MeanStd"),
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
    update_plot("MeanStd")
    layout.start()

