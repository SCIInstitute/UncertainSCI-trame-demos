import os

from itertools import chain, combinations

import numpy as np
import matplotlib.pyplot as plt

from UncertainSCI.distributions import BetaDistribution, ExponentialDistribution
from UncertainSCI.model_examples import laplace_ode_1d, sine_modulation
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion


from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify, observer, matplotlib

import mpld3
from mpld3 import plugins, utils


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
    
    
def build_pce_exp():

    # # Distribution setup

    # Number of parameters
    dimension = 1

    # Specifies exponential distribution
    lbd = 3*np.ones(dimension)
    loc = np.zeros(dimension)
    dist = ExponentialDistribution(lbd=lbd, loc=loc)

    # # Indices setup
    order = 10
    index_set = TotalDegreeSet(dim=dimension, order=order)

    print('This will query the model {0:d} times'.format(index_set.get_indices().shape[0] + 10))
    # Why +10? That's the default for PolynomialChaosExpansion.build_pce_wafp

    # # Initializes a pce object
    pce = PolynomialChaosExpansion(index_set, dist)

    # # Define model
    N = int(1e2)  # Number of degrees of freedom of model
    left = -1.
    right = 1.
    x = np.linspace(left, right, N)
    model = sine_modulation(N=N)

    # # Three equivalent ways to run the PCE model:

    # 1
    # Generate samples and query model in one call:
    pce = PolynomialChaosExpansion(index_set, dist)
    lsq_residuals = pce.build(model)


    # The parameter samples and model evaluations are accessible:
    parameter_samples = pce.samples
    model_evaluations = pce.model_output
    
    
    fig = plt.figure(**figure_size())
    fig.subplots_adjust(hspace=1)
    
    ax1 = fig.add_subplot(4, 1, 1)
    ax1 = mean_stdev_plot(pce, ensemble=50,ax =ax1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2= quantile_plot(pce,ax=ax2, bands=3, xvals = x, xlabel='$x$')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3 = piechart_sensitivity(pce, ax = ax3)
    
    return fig

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
        v_model=("active_figure", "build_pce"),
        items=(
            "figures",
            [
                {"text": "build_pce", "value": "build_pce"},
                {"text": "build_pce_exp", "value": "build_pce_exp"},
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
    update_chart("build_pce")
    layout.start()

