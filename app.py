import pandas as pd
import numpy as np
import itertools
from collections import OrderedDict
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_bootstrap_components as dbc

app = dash.Dash(
)
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.figure_factory as ff

#from diceprobs import get_probs_table, get_probs, pad_cut_probs
from gen_function import roll_dice

app = dash.Dash(__name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            ],
        external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG'])

grid_colors='Greens'

@app.callback(
        Output("the-math", "is_open"),
        [Input("show-math", "n_clicks")],
        [State("the-math", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("odds", "figure"),
    [
        Input("odds-shade", "value"),
        Input("odds-openended", "value"),
    ])
def odds(shade, open_ended):
    if open_ended is None or open_ended == []:
        open_ended = False
    else:
        open_ended = True

    fl = lambda x: list(map(float,x))

    columns = {'obstacle': [i for i in range(10 + 1)]}
    for i in range(1,10):
        column = roll_dice(num_dice=i, shade=shade, open_ended=open_ended, cum_sum=True)
        columns[f"{i}D"] = fl(column)
    df = pd.DataFrame(data=columns).set_index('obstacle')

    #df = get_probs_table(explode_count=explode_count, success_count=success_count)
    df = df.iloc[1:]
    ylabels = [f"Ob {i}" for i in range(1,11)]
    fig = ff.create_annotated_heatmap(z=df.values*100, x=list(df.columns), y=ylabels, colorscale=grid_colors)
    fig.update_layout(
            yaxis_title='Obstacle',
            xaxis = {'showgrid': False},
            yaxis = {'showgrid': False})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.3f}%<extra></extra>",
            xgap=3,
            ygap=3)
    fig['layout']['yaxis']['autorange'] = "reversed"
    for annotation in fig['layout']['annotations']:
        annotation['text'] = "{:.0f}%".format(float(annotation['text']))
    return fig

# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

@app.callback(
    Output("artha", "figure"),
    [
        Input("artha-shade", "value"),
        Input("artha-exponent", "value"),
        Input("artha-openended", "value"),
        Input("artha-options", "value")
    ])
def artha_effect(shade, exponent, open_ended, options):
    if open_ended is None or open_ended == []:
        open_ended = False
    else:
        open_ended = True

    params = OrderedDict()
    params['cum_sum'] = [True]
    params['num_dice'] = [exponent]
    params['open_ended'] = [open_ended]
    if 'aristeia' in options:
        if shade == 'black':
            params['shade'] = ['black', 'grey']
        elif shade == 'grey':
            params['shade'] = ['grey', 'white']
    else:
        params['shade'] = [shade]
        # Can't aristeia white shaded

    if 'divine-inspiration' in options:
        params['divine_inspiration'] = [False, True]
    if 'saving-grace' in options:
        params['saving_grace'] = [False, True]


    if 'boon' in options:
        params['boon'] = [0, 1, 2, 3]
    if 'luck' in options:
        params['luck'] = [False, True]

    # This may be ineffecient, but we must calculate all the dice sets independently anyways
    fl = lambda x: list(map(float,x))
    exact_data = []
    float_data = []
    ycolumns = []
    for values in product_dict(**params):
        artha_cost = [0, 0, 0] #F, P, D
        slug = []
        if values.get('luck', False) == True:
            artha_cost[0] += 1
            slug.append('Luck')
        if values.get('boon', 0) > 0:
            artha_cost[1] += values['boon']
            slug.append(f"+{values['boon']}D Boon")
        if values.get('shade', None) != shade:
            artha_cost[0] += 5
            artha_cost[1] += 3
            artha_cost[2] += 1
            slug.append(f"Aristeia")
        if values.get('divine_inspiration', False):
            artha_cost[2] += 1
            slug.append("Div. Insp.")
        if values.get('saving_grace', False):
            artha_cost[2] += 1
            slug.append("Sav. Gr./C-O")
        slug = ', '.join(slug)
        ycolumns.append(f'{slug} ({artha_cost[0]}F {artha_cost[1]}P {artha_cost[2]}D)')

        exact = roll_dice(**values)
        exact_data.append(exact[1:])
        float_data.append(fl(exact[1:]))

    # Set the first ycolumn as well
    ycolumns[0] = '(0F 0P 0D)'
    height = len(ycolumns)*100/2.5 + 200
    data = np.array(float_data)
    xlabels = [f"Ob {i}" for i in range(1,11)]
    fig = ff.create_annotated_heatmap(z=data*100, x=xlabels, y=ycolumns, colorscale=grid_colors)
    fig.update_layout(
            xaxis_title='Obstacle',
            height=height,
            xaxis = {'showgrid': False},
            yaxis = {'showgrid': False})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.3f}%<br><extra></extra>",
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
rolls and different shades. All these numbers were calculated exactly (not simulated)
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

Source code is freely available (MIT license) at [https://github.com/telnoratti/burningwheel-tools](https://github.com/telnoratti/burningwheel-tools).
            '''),
        dbc.Button(
            "Click here for the math",
            id="show-math",
            className="mb-3",
            color="primary",
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody([
                    dcc.Markdown(r'''
### Generating Functions
To get exact solutions to the probability of a certain number of successes,
this page uses generating functions. [Generating functions](
https://en.wikipedia.org/wiki/Generating_function) can be used for many things,
but essentially it encodes information into a polynomial in order to take
advantage of polynomial algebra (the way the multiply and add to each other).
The method to calculate all the various paths to a certain result is the same
for polynomials. If we can write a polynomial whose coefficients correspond to
the probability of the outcomes of rolling a single die, we can get the
outcomes of multiple dice by multiplying the polynomials.

I solved this more generally, but first consider the case of rolling a B1 test.
That's a test where 4 or above is a success and only one die is rolled. I
modeled this as a two sided die (a coin) with one face 0 and one face 1. When
the test is not open-ended, this is simple,

$$ p_B(x) = \frac{1}{2} + \frac{1}{2}x. $$

The coefficient of $ x^0 $ corresponds to the probability of getting 0 successes, $ \frac{1}{2} $. The coefficient of $x^1$ corresponds to the probability of getting 1 success, $\frac{1}{2}$. In the case of a G1 skill, where 3, 4, 5, and 6 are successes we just have a weighted coin,

$$ p_G(x) = \frac{1}{3} + \frac{2}{3}x. $$

Here success is more likely than failure. If we want to know the odds of rolling G3, we solve for the coefficients of our single die polynomial raised to the number of dice,

$$ p_G(x)^3 = \frac{1}{27} + \frac{2}{9}x + \frac{4}{9}x^2 + \frac{8}{27}x^3. $$

So the odds of getting no successes is $\frac{1}{27}$, one success,
$\frac{2}{9}$, etc. This is a simple calculation with sympy.

    from sympy import Rational, Poly
    from sympy.abc import *

    p_g = Rational("1/3") + Rational("2/3")*x
    print(Poly(p_g**3, x))

The coefficients can be retrieved using the `all_coeffs()` method of the polynomial.

### Exploding dice
Now that we have a method for easily turning any known single die probability
distribution into a multi-die distribution, all we need is the polynomial for a
single exploding die. We were modeling our dice as two sided dice since we
didn't care what the actual roll was, just if it was a success or failure.
We'll need to actually determine the value of the die. To do this we'll add a
couple of additional variables.

$$ \begin{array}{ r l }
s =& \text{probability of a success, depends on shade} \\\\
f =& (1 - s), \text{probability of a failure} \\\\
e =& \text{probability of exploding on a success} \\\\
d =& (1 - e), \text{a dud success}.
\end{array} $$

For an open-ended G1 test, these are $s = \frac{2}{3}$, $f = \frac{1}{3}$, $e = \frac{1}{4}$,
$f = \frac{3}{4}$. Now we need to calculate the odds of getting each number of successes. This is pretty easy with just one die. For any N number of successes, you must have exploded up to N-1 successes. Then you either roll a dud success on the Nth die roll, or explode into a failure. I found this was easier to visualize with a decision tree.

![Decision tree for exploding dice](''' + app.get_asset_url('burningwheel-die-diagram.svg') + r''')

We can build our polynomial with this tree by multiplying every edge we need to take. For zero successes ($x^0$), we get just one option, $f$. For 1 success, we have two different options, $(sd)$ and $(se)f$, we take the sum of these and simplify to get $s(d + ef)$. For 2 successes, we again have two paths, both going through an exploding succes $(se)(sd)$ and $(se)(sef)$. Again take the sum and simplify to get $ses(d + ef)$. We can clearly see a pattern emerge since we need one additional exploding success for N+1 successes. The probability of N successes is $(se)^{n-1}s(d+ef)$. A little bit of index gymnastics gets us $(se)^n\left(\frac{d}{e} + f\right)$. This expression only works for non-zero successes, so our single die polynomial is given by the infinite series,


$$ p\_e(x) = f + \sum\_{n=1}^{\infty} \left( \frac{d}{e} + f \right) (se)^n \, x^n. $$

Let's consider the generating function of an open ended black shade die. This has an equal chance of success and failure and a $\frac{1}{3}$ chance of exploding a success,

$$ \begin{array}{r l}
    p\_{B\_e}(x) &= \frac{1}{2} + \sum\_{n=1}^{\infty} \frac{5}{2}\left(\frac{x}{6}\right)^n, \\\\
    p\_{B\_e}(x) &= \frac{1}{2} + \frac{5}{12}x + \frac{5}{72}x^2 + \frac{5}{432}x^3 + \ldots
\end{array}
$$

    from sympy import Rational, Poly, Order, expand, Sum
    from sympy.abc import *

    # We use the Order term to make the calculations more effecient
    # Increase this to one above the number of successes you want to calculate up to
    p_b_e = Rational("1/2") \
            + (Rational("2/3")/Rational("1/3") + Rational("1/2")) \
            * Sum((Rational("1/2")*Rational("1/3"))**n * x**n, (n, 1, 10)) \
            + O(x**11)
    print(Poly(expand((p_b_e).doit()).removeO()), x)

Note that the sum of the coefficients converges to 1, so we have a discreet
probability distribution. Also the probabilities drop off very quickly
(geometrically), which makes sense as the odds of rolling five 6s in a row are
quite small. Also the odds of rolling 1 success are smaller for open ended dice
than a normal roll ($\frac{5}{12} < \frac{1}{2}$) as some of those successes
are exploding into another success.

As a final demonstration we'll consider the case of three open-ended grey shade
dice (G3). We already calculated our odds above, so we just need to churn numbers.

$$
p\_{G\_e}^3 = \frac{1}{27} + \frac{5}{27}x + \frac{55}{162}x^2 + \frac{815}{2916}x^3 + \ldots
$$

    from sympy import Rational, Poly, O, expand, Sum
    from sympy.abc import *

    p_g_e = Rational("1/3") \
            + (Rational("3/4")/Rational("1/4") + Rational("1/3")) \
            * Sum((Rational("2/3")*Rational("1/4"))**n * x**n, (n, 1, 10)) \
            + O(x**11)
    print(Poly(expand((p_g_e**3).doit()).removeO()), x)

The code used by the site isn't quite as clean as this explanation (yet), but
the calculations are the same.
'''),
                ])
            ),
            id="the-math",
        )
        ],
        style={'maxWidth': 1200, 'border': 'thin lightgrey solid', 'padding': '20px', 'margin': 'auto', 'borderRadius': 5}
    ),
    dbc.Card([
        dbc.CardBody([
            html.H2("Ways Artha Impacts Odds of Success", className="card-title", style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Skill Shade", html_for="artha-shade", width=4),
                        dbc.Col(
                            dbc.Select(
                                id="artha-shade",
                                options=[
                                    {'label': 'Black Shade', 'value': 'black'},
                                    {'label': 'Grey Shade', 'value': 'grey'},
                                    {'label': 'White Shade', 'value': 'white'}],
                                value='black'
                                ),
                            width=8)
                        ], row=True),
                    dbc.FormGroup([
                        dbc.Label("Exponent", width=4),
                        dbc.Col(
                            dbc.Input(id="artha-exponent", type="number", min=0, max=15, step=1, value=3),
                            width=8)
                        ], row=True),
                    dbc.FormGroup([
                        dbc.Label("Open-ended", width=4),
                        dbc.Col(
                            dbc.Checklist(
                                id="artha-openended",
                                options=[
                                    {'label': '', 'value': 'open-ended'}],
                                switch=True),
                            width=8)
                        ], row=True),
                    ]),
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Artha Options"),
                        dbc.Checklist(
                            options=[
                                {'label': 'Luck (1 Fate)', 'value': 'luck'},
                                {'label': 'Boon (1-3 Persona)', 'value': 'boon'},
                                {'label': 'Divine Inspiration (1 Deed)', 'value': 'divine-inspiration'},
                                {'label': 'Saving Grace (1 Deed) / Call-On', 'value': 'saving-grace'},
                                {'label': 'Aristeia (5 Fate, 3 Persona, 1 Deed)', 'value': 'aristeia'},],
                            value=['luck', 'boon', 'divine-inspiration'],
                            id="artha-options"),
                        ])
                    ]),
                ],
                style={'borderBottom': 'thin lightgrey solid', 'padding': '0 0 10px'}),
            dbc.Row([
                ]),
            dbc.Row([
                dbc.Col([
                        dcc.Graph(id="artha")]),
                ]),
            ])
        ],
        style={'margin': '10px 0 0 0'}),
    dbc.Card([
        dbc.CardBody([
            html.H2("Odds of Success", style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("Skill Shade", html_for="artha-shade", width=4),
                        dbc.Col(
                            dbc.Select(
                                id="odds-shade",
                                options=[
                                    {'label': 'Black Shade', 'value': 'black'},
                                    {'label': 'Grey Shade', 'value': 'grey'},
                                    {'label': 'White Shade', 'value': 'white'}],
                                value='black'
                                ),
                            width=8)
                        ], row=True),
                    dbc.FormGroup([
                        dbc.Label("Open-ended", width=4),
                        dbc.Col(
                            dbc.Checklist(
                                id="odds-openended",
                                options=[
                                    {'label': '', 'value': 'open-ended'}],
                                switch=True),
                            width=8)
                        ], row=True),
                    ], width=6),
                dbc.Col(width=2),
                ],
                style={'borderBottom': 'thin lightgrey solid', 'padding': '0 0 10px'}),
            dbc.Row([
                dbc.Col([
                        dcc.Graph(id="odds")
                    ])
                ]),
            ])
        ],
        style={'margin': '10px 0 0 0'}),
    ],
    style={'maxWidth': 1200, 'border': 'thin lightgrey solid', 'padding': 20, 'margin': 'auto', 'borderRadius': 5}
)

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
