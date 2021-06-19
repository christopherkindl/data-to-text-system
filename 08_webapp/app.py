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


# map_zoom = st.selectbox('Zoom level',
#                         ['<select>', 12, 11, 10, 9, 8, 7],
#                         0)  # default value = index 0
# if map_zoom != '<select>':
#         st.map(data, zoom=map_zoom)

stock = st.selectbox(
     'Select a stock to be analysed',
     companies.sp500, help='test')

st.write("")

if stock != "":

    data = yf.Ticker(stock)

    eval_date = data.history(period = 'max', interval = '1d')
    min_date = min(eval_date.index.date)
    max_date = max(eval_date.index.date)

    # year-to-date identification
    end_date = date.today()
    cur_year = date.today().year
    string_date = str(cur_year) + '-01-01'
    start_date = datetime.strptime(string_date, '%Y-%m-%d').date()

    # USER INTERACTION FOR SEARCH
    start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    end_date = st.date_input(label='End date', value=end_date, max_value=max_date)


    tickerDF = data.history(interval = '1d', start = start_date, end = end_date)
    st.write("")

    chart = st.line_chart(tickerDF.Close)



    # st.write('''
    # ### Select time window to narrow down time series analysis''')
    #
    # if start_date > end_date:
    #     # st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    #     st.error('Error: End date must fall after start date.')
    #     st.stop()

    #tickerDF = data.history(period = '1d', start = '2019-01-01', end = '2021-01-01')

    # start_date = ''
    # end_date = ''

    #while type(start_date) and type(end_date) != 'datetime.date':
    # if type(start_date) and type(end_date) != 'datetime.date':
    #
    #     # INITIALE CHART
    #     tickerDF = data.history(period = "ytd", interval = '1d')
    #     chart = st.line_chart(tickerDF.Close)
    #
    # else:
    #
    #     # DYNAMIC CHART BASED ON USER INPUT
    #     tickerDF = data.history(interval = '1d', start = start_date, end = end_date)
    #
    # if not chart:
    #     st.stop()
    #
    # eval_date = data.history(period = 'max', interval = '1d')
    # min_date = min(eval_date.index.date)
    # max_date = max(eval_date.index.date)
    #
    # st.write('''
    # ### Select time window to narrow down time series analysis''')
    #
    # # USER INTERACTION FOR SEARCH
    # start_date = st.date_input(label='Start date', min_value=min_date)
    # end_date = st.date_input(label='End date', max_value=max_date)



# chart = st.line_chart(tickerDF.Close)



# st.write('You selected:', option)

# options = st.multiselect(
#      'Please select a stock',
#      companies.sp500, help='test')
#      # ['Default stock'])

# if not option:
#    st.stop()
#
# st.write('You selected:', option)
# st.write("")
# st.write("")

# ----------------------------------
# Yahoo Finance API & Data Preprocessing
# ----------------------------------

# stock = option
#
# data = yf.Ticker(stock)
#
# tickerDF = data.history(period = '1d', start = '2019-01-01', end = '2021-01-01')
#
# chart = st.line_chart(tickerDF.Close)

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

# ----------------------------------
# Testbed
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

# with st.form("my_form"):
#     st.write("Inside the form")
#     slider_val = st.slider("Form slider")
#     checkbox_val = st.checkbox("Form checkbox")
#
#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         st.write("slider", slider_val, "checkbox", checkbox_val)
#
# st.write("Outside the form")

# min_value = date.fromisoformat('2019-12-01')
# max_value = date.fromisoformat('2019-12-15')
#
# date = st.date_input("Select time window to narrow down time series analysis",
#     min_value=min_value, max_value=max_value)


st.write("")
st.write("")


# st.write('You selected:', options)
