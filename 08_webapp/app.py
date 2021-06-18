import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import companies
import time
from datetime import date

st.write('''
# Summarising time series data of stock exchange data
This app is based on a model that identifies segments by a linear approximatation algorithm and summarises them by natural language''')


options = st.multiselect(
     'Please select a stock',
     companies.sp500, help='test')
     # ['Default stock'])

if not options:
   st.stop()

st.write('You selected:', options[0])
st.write("")
st.write("")

# ----------------------------------
# Yahoo Finance API & Data Preprocessing
# ----------------------------------

stock = options[0]

data = yf.Ticker(stock)

tickerDF = data.history(period = '1d', start = '2019-01-01', end = '2021-01-01')

chart = st.line_chart(tickerDF.Close)

# data = []
# index = []
# for i in range(len(tickerDF.Open)):
#     index.append(tickerDF.Open.index[i])
#     data.append(tickerDF.Open[i])
#     df = pd.DataFrame(data=data, index = index)
#     chart.add_rows(df)
#     time.sleep(0.0025)

# ----------------------------------
# Loading banner
# ----------------------------------

# with st.spinner('Wait for it...'):
#      time.sleep(5)
# st.success('Done!')

# ----------------------------------
# Algorithms
# ----------------------------------

# Linear approximation
# Discrete Fourier Transform
# Multiple Coefficient Binning
# Symbolic Aggregate approximation


# my_chart.add_rows(tickerDF.Open)
#st.line_chart(tickerDF.Volume)
st.write("")

st.write('''
### Summary
Place for output sentences that describe the model's findings.''')

st.write("")

st.write('''
### Under the hood
Algorithms used to identify trends and segments''')

with st.beta_expander("Algorithm 1: Piece-wise Aggregate Approximation"):
     st.write("""
         Explaning how the piecewise linear representation is done.
     """)

with st.beta_expander("Algorithm 2: Discrete Fourier Transform"):
     st.write("""
         Explaning how the piecewise linear representation is done.
     """)

with st.beta_expander("Algorithm 3: Multiple Coefficient Binning"):
     st.write("""
         Explaning how the piecewise linear representation is done.
     """)

with st.beta_expander("Algorithm 4: Symbolic Aggregate approximation"):
     st.write("""
         Explaning how the piecewise linear representation is done.
     """)

# ----------------------------------
# User input: time period
# ----------------------------------

# Connect user input function with max_error_value func

min_value = date.fromisoformat('2019-12-01')
max_value = date.fromisoformat('2019-12-15')

date = st.date_input("Select time window to narrow down time series analysis",
    min_value=min_value, max_value=max_value)


st.write("")
st.write("")



# st.write('You selected:', options)
