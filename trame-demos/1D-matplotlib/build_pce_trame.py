import os

import numpy as np
import matplotlib.pyplot as plt

from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_ode_1d
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify, observer, matplotlib

import mpld3
from mpld3 import plugins, utils


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
    
    fig, ax = plt.subplots(**figure_size())

    ax = mean_stdev_plot(pce, ensemble=50,ax =ax)
    
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


def Subplots():
    fig = plt.figure(**figure_size())
    fig.subplots_adjust(hspace=0.3)

    np.random.seed(0)

    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        color = np.random.random(3)
        ax.plot(np.random.random(30), lw=2, c=color)
        ax.set_title("RGB = ({0:.2f}, {1:.2f}, {2:.2f})".format(*color), size=14)
        ax.grid(color="lightgray", alpha=0.7)

    return fig


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change("active_figure", "figure_size")
def update_chart(active_figure, **kwargs):
    ctrl.update_figure(globals()[active_figure]())


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

layout = SinglePage("Matplotly")
layout.title.set_text("UncertainSCI demo")

with layout.toolbar:
    vuetify.VSpacer()
    vuetify.VSelect(
        v_model=("active_figure", "MeanStd"),
        items=(
            "figures",
            [
                {"text": "Mean and Standard Deviation", "value": "MeanStd"},
                {"text": "Quantile Plot", "value": "Quantiles"},
                {"text": "Sensitivity Piechart", "value": "SensitivityPiechart"},
            ],
        ),
        hide_details=True,
        dense=True,
    )

with layout.content:
    # __properties=[("v_resize", "v-resize:quiet")]
    with vuetify.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
        with observer.SizeObserver("figure_size"):
            html_figure = matplotlib.Figure("figure_0", style="position: absolute")
            ctrl.update_figure = html_figure.update

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    update_chart("MeanStd")
    layout.start()

