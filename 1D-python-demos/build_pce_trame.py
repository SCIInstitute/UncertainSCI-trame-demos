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
    
    ensemble = 50
    
    fig, ax = plt.subplots(**figure_size())
    
    pce.assert_pce_built()
    
    if ensemble:
        ax.plot(x, output[:ensemble, :].T, 'k', alpha=0.8, linewidth=0.2)
    ax.plot(x, mean, 'b', label='Mean')
    ax.fill_between(x, mean-stdev, mean+stdev, interpolate=True, facecolor='red', alpha=0.5, label='$\pm 1$ stdev range')

    #fig = mean_stdev_plot(pce, ensemble=50,**figure_size())
    
    return fig


# -----------------------------------------------------------------------------


def Quantiles():

    bands = 3
    
    xvals = x
    
    xlabel = '$x$'

    pce.assert_pce_built()
    
    dq = 0.5/(bands+1)
    q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
    q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
    quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

    quantiles = pce.quantile(quantile_levels, M=int(2e3))
    median = quantiles[-1, :]

    band_mass = 1/(2*(bands+1))
    
    if ax is None:
        ax = plt.figure().gca()

    ax.plot(x, median, 'b', label='Median')

    for ind in range(bands):
        alpha = (bands-ind) * 1/bands - (1/(2*bands))
        if ind == 0:
            ax.fill_between(x, quantiles[ind, :], quantiles[bands+ind, :],
                             interpolate=True, facecolor='red', alpha=alpha,
                             label='{0:1.2f} probability mass (each band)'.format(band_mass))
        else:
            ax.fill_between(x, quantiles[ind, :], quantiles[bands+ind, :], interpolate=True, facecolor='red', alpha=alpha)

    ax.set_title('Median + quantile bands')
    ax.set_xlabel('$x$')
    ax.legend(loc='lower right')
    
#    fig = quantile_plot(pce, bands=3, xvals=x, xlabel='$x$',**figure_size())

    return fig


# -----------------------------------------------------------------------------


def SensitivityPiechart():
    
    fig = piechart_sensitivity(pce,**figure_size())

    return fig


# -----------------------------------------------------------------------------


def MovingWindowAverage():
    np.random.seed(0)
    t = np.linspace(0, 10, 300)
    x = np.sin(t)
    dx = np.random.normal(0, 0.3, 300)

    kernel = np.ones(25) / 25.0
    x_smooth = np.convolve(x + dx, kernel, mode="same")

    fig, ax = plt.subplots(**figure_size())
    ax.plot(t, x + dx, linestyle="", marker="o", color="black", markersize=3, alpha=0.3)
    ax.plot(t, x_smooth, "-k", lw=3)
    ax.plot(t, x, "--k", lw=3, color="blue")

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
layout.title.set_text("trame ❤️ matplotlib")

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
                {"text": "Moving Window Average", "value": "MovingWindowAverage"},
                {"text": "Subplots", "value": "Subplots"},
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

