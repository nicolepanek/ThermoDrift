# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table


# test function that converts fasta to csv
from temp_model import temp_model

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
title = "ThermoDrift: Predict your protein's stability"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
                       # title & headings
                       html.H1(title),
                       html.H4("Upload your protein FASTA file"),
                       # Upload fasta file
                       dcc.Upload(
                           id='upload-data',
                           children=html.Div([
                                              html.Button('Upload File')
                                              ]),
                           multiple=True
                           ),
                       html.Div(id='output-data-upload')
                       ])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    # decode user uploaded contents
    decoded = base64.b64decode(content_string.split(',')[-1].encode('ascii')).decode()

    # wrap decoded string contents as a stream
    fasta_contents = io.StringIO(decoded)

    # make test csv out of fasta file using test function
    df = temp_model(fasta_contents)

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
                     ])
                     
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)                     
