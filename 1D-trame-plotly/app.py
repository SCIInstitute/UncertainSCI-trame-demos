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
#quantiles = pce.quantile([0.25, 0.5, 0.75]) #  0.25, median, 0.75 quantile
#median = quantiles[-1, :]
    

def MeanStd():

    x_rev = x[::-1]
    upper = mean+stdev
    lower = mean-stdev
    lower=lower[::- 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack((x,x_rev)),
        y=np.hstack((upper,lower)),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Stdev',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean,
        line_color='rgb(0,100,80)',
        name='mean',
    ))

    fig.update_traces(mode='lines')
    
    return fig


# -----------------------------------------------------------------------------


def Quantiles():
    
    bands = 3
    band_mass = 1/(2*(bands+1))
    x_rev = x[::-1]
    
    dq = 0.5/(bands+1)
    q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
    q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
    quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

    quantiles = pce.quantile(quantile_levels, M=int(2e3))
    median = quantiles[-1, :]
    
    fig = go.Figure()
        
    for ind in range(bands):
        alpha = (bands-ind) * 1/bands - (1/(2*bands))
        upper = quantiles[ind, :]
        lower = quantiles[bands+ind, ::-1]
        if ind == 0:
            fig.add_trace(go.Scatter(
                x=np.hstack((x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=True,
                name='{0:1.2f} probability mass (each band)'.format(band_mass),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=np.hstack((x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=False,
            ))

    
    fig.add_trace(go.Scatter(
        x=x, y=mean,
        line_color='rgb(0,0,0)',
        name='median',
    ))

    fig.update_traces(mode='lines')


    return fig


# -----------------------------------------------------------------------------


def SensitivityPiechart():

    global_sensitivity, variable_interactions = pce.global_sensitivity()
    scalarized_GSI = np.mean(global_sensitivity, axis=1)
    print(type(scalarized_GSI))
    labels = [' '.join([pce.plabels[v] for v in varlist]) for varlist in variable_interactions]
    print(type(labels[0]))
    print(variable_interactions)
    print(labels)
    print(scalarized_GSI)

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=scalarized_GSI.tolist())]
        )

#    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
#    values = [4500, 2500, 1053, 500]
#
#    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])


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


