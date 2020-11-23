import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_bootstrap_components as dbc

app = dash.Dash(
)
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff

from diceprobs import get_probs_table, get_probs, pad_cut_probs

app = dash.Dash(__name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            "https://codepen.io/chriddyp/pen/brPBPO.css"
            ],
        external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG'])

grid_colors='Greens'

@app.callback(
    Output("standard", "figure"),
    [
        Input("open-ended", "on"),
        Input("shade", "value"),
    ])
def ten_by_ten(open_ended, success_count):
    explode_count = 0
    if open_ended:
        explode_count = 1
    df = get_probs_table(explode_count=explode_count, success_count=success_count)
    df = df.iloc[1:]
    ylabels = [f"Ob {i}" for i in range(1,11)]
    fig = ff.create_annotated_heatmap(z=df.values*100, x=list(df.columns), y=ylabels, colorscale=grid_colors)
    fig.update_layout(
            yaxis_title='Obstacle',
            xaxis = {'showgrid': False},
            yaxis = {'showgrid': False})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.0f}%<extra></extra>",
            xgap=3,
            ygap=3)
    fig['layout']['yaxis']['autorange'] = "reversed"
    for annotation in fig['layout']['annotations']:
        annotation['text'] = "{:.0f}%".format(float(annotation['text']))
    return fig

@app.callback(
    Output("artha", "figure"),
    [
        Input("artha-shade", "value"),
        Input("skill", "value")
    ])
def artha_effect(success_count, skill):
    # This may be ineffecient, but we must calculate all the dice sets independently anyways
    data = []
    ycolumns = []
    # Divine inspiratoin
    for die_mult in [1, 2]:
        # epiphany
        shades = [success_count]
        if success_count < 5:
            shades = [success_count, success_count + 1]
        # Minor Epiphany (Aristeia)
        for success_count_l in shades:
            # Open-ended either from Fate with Luck or magic
            for explode_count in [0, 1]:
                # Boon from persona or maybe forks
                for boon in [0, 1, 2, 3]:
                    slug = []
                    if explode_count == 1:
                        slug.append('Open-ended (Fate)')
                    if boon > 0:
                        slug.append(f"+{boon}D (Persona)")
                    if success_count_l != success_count:
                        slug.append(f"Aristeia")
                    if die_mult > 1:
                        slug.append("Divine Inspiration (Deed)")
                    slug = ', '.join(slug)
                    ycolumns.append(slug)

                    num_dice = die_mult*skill + boon
                    data.append(pad_cut_probs(get_probs(num_dice=num_dice, explode_count=explode_count, success_count=success_count_l), 11)[1:])

    # Set the first ycolumn as well
    ycolumns[0] = 'Basic'
    data = np.array(data)
    xlabels = [f"Ob {i}" for i in range(1,11)]
    fig = ff.create_annotated_heatmap(z=data*100, x=xlabels, y=ycolumns, colorscale=grid_colors)
    fig.update_layout(
            xaxis_title='Obstacle',
            height=800,
            xaxis = {'showgrid': False},
            yaxis = {'showgrid': False})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.0f}%<extra></extra>",
            xgap=3,
            ygap=3)
    fig['layout']['yaxis']['autorange'] = "reversed"
    for annotation in fig['layout']['annotations']:
        annotation['text'] = "{:.0f}%".format(float(annotation['text']))
    return fig


# At some point I want to get the generating function rendered as latex
# $$ p(x) = \frac{1}{2}x^0 + \sum_{n=1}^{\infty} \frac{5}{2}\left(\frac{1}{6}\right)^n x^n $$
app.layout = html.Div([
    html.Div([
        dcc.Markdown('''
# About The Page

This is a dice roll probability analysis for the game [Burning
Wheel](https://www.burningwheel.com/) by [Luke Crane](https://twitter.com/burning_luke?lang=en). The game uses a few different
dice mechanics that deviate from most TTRPG dice rolling. You roll a pool of
six-sided dice, counting 4, 5, and 6 as a "success" in order to meet a set
obstacle, (i.e. a roll of 2, 3, 5, 6 would succeed against obstacle 1 and
obstacle 2). Some skills are "open-ended" which means you roll an additional
die for every 6 rolled. During play you gain resources called Artha which can
be spent in order to manipulate the dice. This page is designed to help players
and the GM how spending those resources will impact the odds of success.

There are two charts on this page, the first calculates the impact of various
ways to spend Artha on the odds of success. The second is a more general graph
comparing different die pools and obstacles. Both allow exploring open-ended
rolls and different shades. All these numbers were calculated analytically
using [sympy](https://www.sympy.org/). I dabbled with simulating the dice rolls
with [AnyDice](https://anydice.com/), but was not satisfied with the results.
Doing it myself led to ineffecient simulations that took too long to be
interactive, so I ended up blowing a weekend going back through my old
combinatorics book.

Credit goes largely to the author of [this page on Firestorm Armada](https://www3.risc.jku.at/education/courses/ws2016/cas/exploding.html)
for doing a very similar, but slightly more complex problem and helping scrape
off the cruft around generating functions in my mind. Also Dean Baker who put
together some wonderful documents on [his site](http://customrpgfiles.wikidot.com/burning-wheel).

I would like to also include how call-on skills impact dice rolls, though that is a post-roll mechanic.
            '''),
        ],
        style={'max-width': 1200, 'border': 'thin lightgrey solid', 'padding': '20px', 'margin': 'auto', 'border-radius': 5}
    ),
    html.Div([
        html.H2("Ways Artha Impacts Odds of Success", style={'text-align': 'center'}),
        dbc.Row([
            dbc.Col([
                html.P("Skill Shade"),
                dcc.Dropdown(
                    id="artha-shade",
                    options=[
                        {'label': 'Black Shade', 'value': 3},
                        {'label': 'Grey Shade', 'value': 4},
                        {'label': 'White Shade', 'value': 5},
                    ],
                    value=3,
                )]
            ),
            dbc.Col([
                html.P("Skill Exponent"),
                daq.NumericInput(
                    id='skill',
                    value=3
                )]
            )],
            style={'borderBottom': 'thin lightgrey solid', 'padding': '0 0 10px'}
        ),
        dbc.Row([
            dbc.Col([
                #dcc.Loading(
                #    id="loading-artha",
                #    type="graph",
                #    children=[
                        dcc.Graph(id="artha")
                #        ],
                #    )
                ]),
            ]),
        ],
        style={'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px'}
    ),
    html.Div([
        html.H2("Odds of Success", style={'text-align': 'center'}),
        dbc.Row([
            dbc.Col([
                html.P("Open-ended skill"),
                daq.BooleanSwitch(
                    id="open-ended",
                    on=False,
                )]
            ),
            dbc.Col([
                html.P("Skill Shade"),
                dcc.Dropdown(
                    id="shade",
                    options=[
                        {'label': 'Black Shade', 'value': 3},
                        {'label': 'Grey Shade', 'value': 4},
                        {'label': 'White Shade', 'value': 5},
                    ],
                    value=3,
                )]
            )],
            style={'borderBottom': 'thin lightgrey solid', 'padding': '0 0 10px'}
        ),
        dbc.Row([
            dbc.Col([
                #dcc.Loading(
                #    id="loading-standard",
                #    type="default",
                #    children=[
                        dcc.Graph(id="standard")
                #        ],
                #    )
                ])
            ]),
    ])],
    style={'max-width': 1200, 'border': 'thin lightgrey solid', 'padding': 20, 'margin': 'auto', 'border-radius': 5}
)

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
