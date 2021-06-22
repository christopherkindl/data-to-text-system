import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import companies
import time
from datetime import date
from datetime import datetime, timedelta
import segment
import wrappers
from test_list import list_segments # import test segments
import fit
import math
#from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
#from matplotlib.lines import Line2D

# ----------------------------------
# Functions
# ----------------------------------

def segment_detection(df, max_error):

    # identify index integer based on selected start and end date
    start_date = list(df.index)[0]
    start = df.index.get_loc(start_date)
    end_date = list(df.index)[-1]
    end = df.index.get_loc(end_date)

    # transform input date from datetime to string date
    #start_date = start_date.strftime('%Y-%m-%d')
    #end_date = end_date.strftime('%Y-%m-%d')
    #start_date
    #end_date

    # get index integer of date index
    #start = df.index.get_loc(start_date)
    #end = df.index.get_loc(end_date)

    # create list out of df column
    series = list(round(df.Close, 2))

    # detect segments and filter series based on selected time range
    segments = segment.bottomupsegment(series[start:end], fit.interpolate, fit.sumsquared_error, max_error)

    return segments

def index_to_date(df, segments):
    dates = []
    index = list(df.index)

    for segment in segments:
        dates.append(index[segment[0]].date())

    return dates

def max_error_value(series):

    min_value = math.floor(min(series))
    max_value = math.floor(max(series))
    range_value = max_value - min_value

    error_rate = 0.3 * max_value

    return error_rate

# def draw_segments(segments):
#     ax = gca()
#     for segment in segments:
#         line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
#         ax.add_line(line)

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

    # TEST TEST ----

    # df = data.history(period = 'max', interval = '1d')

    # series = list(round(df['Close'], 2))

    # ------

    # initialise date range with year-to-date


    end_date = date.today() - timedelta(days=1)
    cur_year = date.today().year
    string_date = str(cur_year) + '-01-04'

    # end_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    # start_date = datetime.strptime(max_date, '%Y-%m-%d').date()

    # transform string date to datetime format
    start_date = datetime.strptime(string_date, '%Y-%m-%d').date()

    st.write('''
    Select a time period to narrow down the analysis''')

    # user field for date selection
    start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    end_date = st.date_input(label='End date', value=end_date, max_value=max_date)

    # VALIDATION
    # weekday1 = start_date.weekday()
    # weekday1
    # weekday2 = end_date.weekday()
    # weekday2

    # check whether selected dates are weekend days
    if start_date.weekday() >= 5 or end_date.weekday() >= 5:
        st.warning('Please select weekdays')
        st.stop()

    # get stock data filtered by selected date range
    tickerDF = data.history(interval = '1d', start = start_date, end = end_date)
    length = len(tickerDF)


    # identify max error
    series = list(round(tickerDF.Close, 2))
    max_error = max_error_value(series)
    st.write('Max error rate based on selected time window: ', max_error)

    # # TEST -----
    # a = list(tickerDF.index)[-1]
    # a

    st.write("")

    # algos = ['dog', 'cat', 'fish']
    # pet = st.radio('Pick an algorithm', algos)

    # display chart
    chart = st.line_chart(tickerDF.Close)


    # time series decomposition

    # assign DF
    # df = tickerDF

    # start_date = list(df.index)[0].date()
    # end_date = list(df.index)[-1].date()

    # assign segments
    segments = segment_detection(tickerDF, max_error)

    cur_index = []
    cur_data = []

    for i in range(len(segments)):
            cur_index.append(list(tickerDF.index)[segments[i][0]])
            cur_data.append(segments[i][1])
            df_test = pd.DataFrame(data=cur_data, index = cur_index)
            chart.add_rows(df_test)
            time.sleep(0.5)



    # cur_data = []
    # cur_index = []
    #
    # for i in range(len(tickerDF.Open)):
    #     cur_index.append(tickerDF.Open.index[i])
    #     cur_data.append(tickerDF.Open[i])
    #     df = pd.DataFrame(data=cur_data, index = cur_index)
    #     chart.add_rows(df)
    #     time.sleep(0.0025)


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
