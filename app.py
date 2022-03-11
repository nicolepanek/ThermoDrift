# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
from dash import dash_table
#from temp_model import temp_model 
import pandas as pd


app = Dash(__name__)

title="ThermoDrift: Predict your protein's stability"

app.layout = html.Div([
                       # GUI headings
                       html.H1(title),
                       html.H4('Upload your protein FASTA File'),

                       # Upload fasta file(s)
                       dcc.Upload(
                           id='upload-data',
                           children=html.Div([
                                              html.Button('Upload File')
                                              ]),
                           ),
                       html.Div(id='data-table')
                       ])
   
@app.callback(
    Output(component_id='data-table', component_property='children'),
    Input(component_id='upload-data', component_property='contents')
    )

# Process uploaded fasta file(s)
def process_fasta(contents):

#    df = temp_model(contents)
    children='hello World!'
    
    return children
              
#app.layout = dash_table.DataTable(df)


if __name__ == '__main__':
    app.run_server(debug=True)
