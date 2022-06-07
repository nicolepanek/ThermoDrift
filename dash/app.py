# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io
import dash
import pandas as pd

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
#from temp_model import temp_model
from model.CNN_model import thermodrift_model
from model.CNN_model.inference_script import main




# list of one letter amino acid codes
list_aa = list("ARNDCQEGHILKMFPSTWYVUX_?-")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
title = "ThermoDrift: Predict protein stability"
logo_filename = "images/thermodrift_logo.png"
encoded_logo = base64.b64encode(open(logo_filename, 'rb').read())

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div([# title & headings & logo
                       html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), height=300),
                       html.H1(title),
                       # Tabs
                       dcc.Tabs(
                        id='tabs-all',
                        value='about-thermodrift',
                        parent_className='custom-tabs',
                        className='custom-tabs-container',
                        children=[
                            dcc.Tab(
                                label='About ThermoDrift',
                                value='about-thermodrift',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                ),
                            dcc.Tab(
                                label='Sequence Prediction',
                                value='sequence-prediction',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                ),
                            ]),
                       html.Div(id='tabs-content')

                       ])
                       
# display tabs
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-all', 'value'))


def render_content(tab):
    if tab == 'about-thermodrift':
        return html.Div([
            html.H4('What is ThermoDrift?'),
            html.P(
                'ThermoDrift is a user friendly tool to classify '
                'protein sequences as Thermophilic, Mesophilic, or Psychrophilic. '
                'This tool can be used for prediction, but is also an open ended '
                'tool for people to play with and adapt for the tasks they need. '
                'Look at the figure below to see what temperatures these organisms '
                'live at.'),
            html.P(
                'Thermodrift was created as an open source project that enables '
                'automated protein classification for thermophilic, mesophilic, or '
                'psychrophilic phenotypes to address the lack of any computational '
                'classifiers for thermostable protein prediction that are widely '
                'accessible and cater to a scientific user base with little machine '
                'learning experience but a lot of enthusiasm for protein characterization.')
            ])
    elif tab == 'sequence-prediction':
        return html.Div([
            html.H4("Upload protein FASTA File"),
            # Upload fasta file
            html.Hr(),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Button('Upload FASTA File')
                    ]),
                # allow multiple files to be uploaded
                multiple=True),
            html.Div(id='output-data-upload'),
            # Download csv of output data
            html.Hr(),
            html.Button('Download CSV', id='btn_csv'),
            dcc.Download(id='download-dataframe-csv')

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

            global df 
            df = main(fasta_contents)
            df = df[['protein',
            'prediction',
            'thermophile probability',
            'mesophile probability',
            'psychrophile probability',
            'sequence']]

            return html.Div([
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),
                dash_table.DataTable(df.to_dict('records'),
                    [{'name': i, 'id': i} for i in df.columns],
                    style_cell={'textAlign': 'left',
                    'padding': '5px'},
                    style_header={'backgroundColor': 'white',
                    'fontWeight': 'bold'}
                    ),
                html.Hr(), # horizontal line
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



# upload fasta file
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    suppress_callback_exceptions=True
    )

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    return children

# download csv
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
    suppress_callback_exceptions=True
    )
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "thermodrift_output.csv")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)                     
