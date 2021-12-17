#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install jupyter-dash --user


# In[5]:


import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import plotly.io as pio

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


plt.rcParams["figure.figsize"] = (20,3)

# prep strava data
strava_data2 = pd.read_csv("Strava_readings.csv")
strava_data2['Date'] = pd.to_datetime(strava_data2['Date'])
strava_data2 = strava_data2.resample('H', on='Date').sum()

# prep light data
light_intensity2 = pd.read_csv("Photosensor_readings.csv")
light_intensity2['Date'] = pd.to_datetime(light_intensity2['Date'])
light_intensity2 = light_intensity2.resample('H', on='Date').mean()
#print(light_intensity2)

dataframe = light_intensity2.copy() # copy the dataframe to merge the two
#dataframe.dropna() # this removes all hours without light intensity - use instead of the ffill() command below
dataframe['Kudos'] = strava_data2['Kudos']
dataframe['Distance'] = strava_data2['Distance']
dataframe['Pace'] = strava_data2['Pace']
dataframe['Elapsed time'] = strava_data2['Elapsed time']

# use this code to forward fill light intensities to times that don't have them, instead of the dropna() command above
dataframe['Light Intensity'] = dataframe['Light Intensity'].ffill()
dataframe = dataframe.fillna(0) # this replaces empty strava data with 0s

# Load Data
df = dataframe
#print(df)

tabs_styles = {'zIndex': 99, 'display': 'inlineBlock', 'height': '4vh', 'width': '12vw',
               'position': 'fixed', "background": "#323130", 'top': '12.5vh', 'left': '7.5vw',
               'border': 'grey', 'border-radius': '4px'}


# Build App
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)


colors = {
    'background': '#FFFFFF',
    'text': '#0047ab'
}


fig1 = px.line(df, y=["Light Intensity", "Distance"])
fig2 = px.line(df, y=["Light Intensity", "Elapsed time"])


#fig.update_layout(
#    plot_bgcolor=colors['background'],
#    paper_bgcolor=colors['background'],
 #   font_color=colors['text']

#)


app.layout = html.Div(


    style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Sensing & IOT - coursework 2',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family': 'sans-serif'
        }  
    ),

    html.Div(children='Can we correlate light intensity with features of my Strava Actvities?', 
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family': 'sans-serif'
    }),
    
    dcc.Tabs([
        dcc.Tab(label='Light Intensity and Distance',   
                style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family': 'sans-serif'
        },
                children=[
            
            dcc.Graph(
                id = "Light intensity distance graph",
          
                figure=fig1
                     
            )
           
    
        ]),
    
     dcc.Tab(label='Light Intensity and Elapsed Time', 
             style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family': 'sans-serif'
        },
             children=[
            
        dcc.Graph(
            id = "Light intensity elapsed time graph",
            
            figure=fig2,
            
            
                     
            )
        ]),
        
       dcc.Tab(label='Data actuation', 
             style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family': 'sans-serif'
        },
             children=[
            
      html.Div([
        dcc.Input(id='input-1-state', type='text', value='How light is it outside?'),
        dcc.Input(id='input-2-state', type='text', value='How are you feeling?'),
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
        html.Div(id='output-state')
        
])
        ]),
        
   
]) 
    
   
])
      
@app.callback(Output('output-state', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('input-1-state', 'value'),
              State('input-2-state', 'value'))
def update_output(n_clicks, input1, input2):
    if n_clicks ==1:
        return u'''
            It is "{}" outside,
            you feel "{}",
            And you have pressed this button {} times, which means you should probably go for a run
        
            '''.format(input1, input2,n_clicks)
       
        n_clicks = 0
   

# Run app and display result inline in the notebook
app.run_server(mode='external', debug=True) #to run locally use mode="inline" and to run externally use mode="external"


# In[ ]:




