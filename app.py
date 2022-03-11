# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd


app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='ThermoDrift'),

    html.Div(children='''
        ThermoDrift: Predict your proteins stability!
    ''')
                                ])



                                                                              

if __name__ == '__main__':
    app.run_server(debug=True)
