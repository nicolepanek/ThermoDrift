# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io
import dash
import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import dcc, html, dash_table

#from temp_model import temp_model
from model.CNN_model.inference_script import main

# list of one letter amino acid codes
list_aa = list("ARNDCQEGHILKMFPSTWYVUX_?-")
content_style = {"margin-left": "10rem", "margin-right": "16rem", "padding": "2rem 1rem"}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## FILE HANDLING 
# Logo
logo_filename = "./images/thermodrift_logo.png"
encoded_logo = base64.b64encode(open(logo_filename, 'rb').read())
# background info
bg_img = "./images/thermo_classes.png"
encoded_bg = base64.b64encode(open(bg_img, 'rb').read())
annot_img = "./images/thermo.png"
encoded_annot = base64.b64encode(open(annot_img, 'rb').read())

# Model architecture
architecture_img = "./images/model_architecture.png"
encoded_arch = base64.b64encode(open(architecture_img, 'rb').read())
# Training curves
training_img = "./images/training_curve_all_models.png"
encoded_training = base64.b64encode(open(training_img, 'rb').read())
# Confusion matrices
confusion_1 = "./images/220606_fig1_CNNmodel_analysis_heatmaponly.png"
encoded_confusion1 = base64.b64encode(open(confusion_1, 'rb').read())
confusion_2 = "./images/220606_fig2_esm1b_classifier_heatmaponly.png"
encoded_confusion2 = base64.b64encode(open(confusion_2, 'rb').read())
# tsne
tsne = "./images/tsne.png"
encoded_tsne = base64.b64encode(open(tsne, 'rb').read())


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(children=[# title & headings & logo
                       html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), height=300),
                       # Tabs
                       dcc.Tabs(
                        id='tabs-all',
                        value='about-thermodrift',
                        parent_className='custom-tabs',
                        className='custom-tabs-container',
                        children=[
                            dcc.Tab(
                                label='About',
                                value='about-thermodrift',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                                ),
                            dcc.Tab(
                                label='Model Architecture',
                                value='model-architecture',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                ),
                            dcc.Tab(
                                label='Sequence Prediction',
                                value='sequence-prediction',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                )
                            ]),
                       html.Div(id='tabs-content')

                       ])
                       
# display tabs
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-all', 'value'))


def render_content(tab):
    if tab == 'about-thermodrift':
        return html.Div(style=content_style, children=[
            html.H4('What is ThermoDrift?'),
            html.P(
                'Proteins store cryptic information about their stability in different thermal environments '
                '(psychrophilic, mesophilic, and thermophilic) within their sequences. '
                'ThermoDrift uses a machine learning model that identifies these patterns in protein '
                'sequences (described below) to predict their thermal class.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_bg.decode()), height=250),
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
                'learning experience but a lot of enthusiasm for protein characterization.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_annot.decode()), height=250),
            html.H4("References"),
            html.P(
                'Metpally, R.P.R., Reddy, B.V.B. Comparative proteome analysis of psychrophilic ' 
                'versus mesophilic bacterial species: Insights into the molecular basis of cold '
                'adaptation of proteins. BMC Genomics 10, 11 (2009). https://doi.org/10.1186/1471-2164-10-11'
                ),
            html.P(
                'Lieph, R., Veloso, F.A., and Holmes, D.S. (2006). Thermophiles like hot T. Trends Microbiol. '
                '14, 423–426.'
                ),
            html.P(
                'Alexander, R., Joshua, M., Tom, S., Siddharth, G., Zeming, L., Jason, L., Demi, G., Myle, O., '
                'Lawrence, Z.C., Jerry, M., et al. (2021). Biological structure and function emerge from scaling '
                'unsupervised learning to 250 million protein sequences. Proc. Natl. Acad. Sci. 118, e2016239118.'
                )
            ])
    if tab == 'model-architecture':
        return html.Div(style=content_style, children=[
            html.H4("Model Architecture"),
            html.P(
                'Our best model architecture used protein embeddings from the pretrained '
                'ESM1b transformer fed into a fully connected linear classifier.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_arch.decode()), height=250),
            html.H4("Model Performance"),
            html.P(
                'The ESM1b classifier’s cross entropy loss decreases across epochs '
                'while that of the CNN model remains stagnant.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_training.decode()), height=400),
            html.P(
                'Confusion matrices exhibit the improved accuracy of thermal class prediction '
                'by the ESM1b classifier as compared to the CNN model.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_confusion1.decode()), height=300),
            html.Img(src='data:image/png;base64,{}'.format(encoded_confusion2.decode()), height=300),
            html.P(
                'While there are clusters that do form for the same class after the embeddings, ' 
                'there is still a lot of noise and room for improvement in the future.'
                ),
            html.Img(src='data:image/png;base64,{}'.format(encoded_tsne.decode()), height=300),
            ]),

    if tab == 'sequence-prediction':
        return html.Div(style=content_style, children=[
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
            df_to_display = df[['protein',
            'prediction',
            'thermophile probability',
            'mesophile probability',
            'psychrophile probability']]

            return html.Div(children=[
                html.P(f"Processed source file: {filename}"),
                html.P(f"Timestamp: {datetime.datetime.fromtimestamp(date)}"),
                dash_table.DataTable(df_to_display.to_dict('records'),
                    [{'name': i, 'id': i} for i in df_to_display.columns],
                    style_cell={'textAlign': 'left',
                    'padding': '5px'},
                    style_header={'backgroundColor': 'white',
                    'fontWeight': 'bold'}
                    )
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
    app.run_server(debug=False, port=8050)                     
