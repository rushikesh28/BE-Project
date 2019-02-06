# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:16:25 2018

@author: rushi
"""

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from pprint import pprint as p

import pandas as pd
plotly.tools.set_credentials_file(username='rushi123', api_key='UCyFpAIMKuaeuMV9KqcG')
df = pd.read_csv('C:\\Users\\rushi\\Desktop\\BE Project\\All datasets combined.csv')

#data = [go.Scatter(x=df.Date, y=df.High)]

#rows = df.values.tolist()  # convert dataframe into a list
#rows.reverse()

"""plotly.offline.plot({
        "data": [go.Scatter(x=df.Year, y=df.TotalNeonataldeaths)],
                "layout":go.Layout(title="Neonatal Mortality") })"""
                
"""plotly.offline.plot({
        "data": [go.Scatter(x=df.Year, y=df.TotalPostNeonataldeaths)],
                "layout":go.Layout(title="Post Neonatal Mortality") })"""

"""plotly.offline.plot({
        "data": [go.Scatter(x=df.Year, y=df.PostneonataldeathsduetoMalaria)],
                "layout":go.Layout(title="Post Neonatal deaths due to Malaria") })"""

"""plotly.offline.plot({
        "data": [go.Scatter(x=df.Year, y=df.Postneonataldeathsduetodiarrhoea)],
                "layout":go.Layout(title="Post Neonatal deaths due to diarrhoea") })"""

plotly.offline.plot({
        "data": [go.Scatter(x=df.Year, y=df.PostneonataldeathsduetoAcuteRespiratoryInfection)],
                "layout":go.Layout(title="Post Neonatal deaths due to acute respiratory infection") })


