import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import companies
import time
from datetime import date
from datetime import datetime

st.write('''
# Summarising time series data of stock exchange data
This app is based on a model that identifies segments by a linear approximatation algorithm and summarises them by natural language''')


stock = st.selectbox(
     'Select a stock to be analysed',
     companies.sp500, help='test')

st.write("")

if stock != "Search for company share code":

    # get stock data of selected company
    data = yf.Ticker(stock)

    # identify max and min date of selected stock
    eval_date = data.history(period = 'max', interval = '1d')
    min_date = min(eval_date.index.date)
    max_date = max(eval_date.index.date)

    # initialise date range with year-to-date
    end_date = date.today()
    cur_year = date.today().year
    string_date = str(cur_year) + '-01-01'

    # transform string date to datetime format
    start_date = datetime.strptime(string_date, '%Y-%m-%d').date()

    st.write('''
    Select a time period to narrow down the analysis''')

    # user field for date selection
    start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    end_date = st.date_input(label='End date', value=end_date, max_value=max_date)

    # get stock data filtered by selected date range
    tickerDF = data.history(interval = '1d', start = start_date, end = end_date)
    st.write("")

    # display chart
    chart = st.line_chart(tickerDF.Close)


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


st.write("")
st.write("")
