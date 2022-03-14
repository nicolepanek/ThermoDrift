# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io
import dash
import pandas as pd

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
# dummy function for temporary model
from temp_model import temp_model
# list of one letter amino acid codes
list_aa = list("ARNDCQEGHILKMFPSTWYVUX_?-")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
title = "ThermoDrift: Predict your protein's stability"
heading1 = "Upload your protein FASTA File" 

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
                       # title & headings
                       html.H1(title),
                       html.H4(heading1),
                       # Upload fasta file
                       dcc.Upload(
                           id='upload-data',
                           children=html.Div([
                                              html.Button('Upload FASTA File')
                                              ]),
                           multiple=True
                           ),
                       html.Div(id='output-data-upload'),
                       #Button to download .csv of output data
                       html.Button("Download CSV", id="btn_csv"),
                       dcc.Download(id="download-dataframe-csv"),
                       ])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    # check in user uploads a fasta file
    if 'fasta' in filename:
        # decode user uploaded fasta contents
        decoded = base64.b64decode(content_string.split(',')[-1].encode('ascii')).decode()

        # check if sequence has characters not within amino acid one-letter codes
        fasta_seq =list(decoded.split('\n')[1])
        if all(x in list_aa for x in fasta_seq) == False:
            return html.Div(["""
                              Error: Fasta protein sequence contains character not found in one-letter amino acid codes.
                                     Please format protein sequence in fasta file with one-letter amino acid codes.
                             """])
        else:
            # wrap decoded string contents as a stream
            fasta_contents = io.StringIO(decoded)
            # call our dummy function
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
                             html.Pre(contents[0:200] + '...',
                                      style={
                                          'whiteSpace': 'pre-wrap',
                                          'wordBreak': 'break-all'
                                          })
                             ])
                               
                             
    else:
        return html.Div(["""
                         Error: Wrong file type uploaded. 
                                Please upload a FASTA file.
                         """])
df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 5, 6], "c": ["x", "x", "y", "y"]})        

                     
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

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "mydf.csv")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)                     
