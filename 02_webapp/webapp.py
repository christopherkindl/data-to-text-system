# standard libraries
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer
import math
from math import floor
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from numpy.linalg import lstsq
from statistics import mean

# webapp framework
import streamlit as st

# yahoo finance API and S&P 500 companies
import yfinance as yf
import companies

# preprocessing, fuzzy sets and segmentation
import fuzzy
import preprocessing

# change point detection algorithms
import ruptures as rpt

# ----------------------------------
# Functions
# ----------------------------------

def leastsquareslinefit(sequence,seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""

    x = np.arange(seq_range[0],seq_range[1]+1)
    y = np.array(sequence[seq_range[0]:seq_range[1]+1])
    A = np.ones((len(x),2),float)
    A[:,0] = x
    (p,residuals,rank,s) = lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p,error)

def sumsquared_error(sequence, segments):
    '''Return the sum of squared errors for a least squares line fit of one segment of a sequence'''

    total_error = []

    for segment in segments:
        x0,y0,x1,y1 = segment
        p, error = leastsquareslinefit(sequence,(x0,x1))
        total_error.append(round(error, 3))

    sumsquared_error = round(sum(total_error), 2)

    return sumsquared_error

def meansquared_error(sequence, segments):
    '''Return the mean of squared errors for a least squares line fit of one segment of a sequence'''

    total_error = []

    for segment in segments:
        x0,y0,x1,y1 = segment
        p, error = leastsquareslinefit(sequence,(x0,x1))
        total_error.append(error)

    meansquared_error = round(mean(total_error), 2)

    return meansquared_error

def suffix(d):
    '''Return date in customised string format'''
    return 'th' if 11<=d<=13 else {1:'st',2:'nd',3:'rd'}.get(d%10, 'th')

def custom_strftime(t):
    '''Return date in customised string format'''
    format = '%B {S}, %Y'
    string_date = t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))

    return string_date

def max_price(df):
    '''Return highest price of a series'''

    max_value = 0
    index = None

    for i, value in enumerate(df['Close Price']):
        if value > max_value:
            max_value = round(value, 2)
            index = df['Close Price'].index[i].date()
            string_date = custom_strftime(index)

    return max_value, string_date

def min_price(df):
    '''Return lowest price of a series'''

    min_value = 999999
    index = 0

    for i, value in enumerate(df['Close Price']):
        if value < min_value:
            min_value = round(value, 2)
            index = df['Close Price'].index[i].date()
            string_date = custom_strftime(index)

    return min_value, string_date

def min_date(df):
    '''Return corresponding date to the minimum price of series'''
    min_date = custom_strftime(df.index.min())

    return min_date

def max_date(df):
    '''Return corresponding date to the maximum price of series'''

    max_date = custom_strftime(df.index.max())

    return max_date

def summary_price(df):
    '''Template for general information sentence. Includes information of highest and lowest price of a series'''

    opening = 'During the time period of '
    #mindate = min_date(df)
    #maxdate = max_date(df)
    glue = ' and '
    date_element = ' the stock price peaked at '
    max_value = max_price(df)[0]
    glue_2 = ' on '
    date_1 = max_price(df)[1]
    glue_3 = ' and hit its lowest value of '
    min_value = min_price(df)[0]
    glue_4 = ' on '
    date_2 = min_price(df)[1]
    eos = '.'

    sentence0 = opening + str(custom_strftime(df.index.min().date())) + glue + str(custom_strftime(df.index.max().date())) + date_element + str(max_value) + glue_2 + date_1 + glue_3 \
                + str(min_value) + glue_4 + date_2 + eos

    return sentence0


def create_segments(breaks, df):
    '''
    1. Input change point information and df.
    2. Return segments.
    '''
    segments = []
    start = df.index[0]
    index = list(df.index)
    for breakpoint in breaks:
        segment = (index.index(start), round(df["Open"][start], 4), index.index(breakpoint), round(df["Open"][breakpoint], 4))
        segments.append(segment)
        start = breakpoint
    if df.index[-1] != breaks[-1]:
        final = df.index[-1]
        segment = (index.index(breaks[-1]), round(df["Open"][breaks[-1]], 4), index.index(final), round(df["Open"][final], 4))
        segments.append(segment)
    return segments

def changepoint_detection(algorithm, series):
    '''
    1. Input change point detection algorithm and series.
    2. Detect change points
    3. Return detected change points and elapsed time.
    '''

    if algorithm == 'Dynamic Programming (Optimal Method)':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.Dynp(model="l1").fit(series)

        # detect change point
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change point
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)


    if algorithm == 'Bottom-Up':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.BottomUp(model="l1").fit(series)

        # detect change points
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change points
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Window Sliding (40)':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.Window(width=40, model="l1").fit(series)

        # detect change points
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change points
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Window Sliding (60)':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.Window(width=60, model="l1").fit(series)

        # detect change points
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change points
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Window Sliding (100)':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.Window(width=100, model="l1").fit(series)

        # detect change points
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change points
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

        if algorithm == 'Window Sliding (200)':

            # create model with least absolute deviation cost function
            start = timer()
            model = rpt.Window(width=200, model="l1").fit(series)

            # detect change points
            breaks_pp = model.predict(n_bkps=n_breaks-1)
            end = timer()
            elapsed_time = round(end - start, 2)

            # get corresponding date for change points
            breaks = []
            for i in breaks_pp:
                breaks.append(ts.index[i-1])
            breaks = pd.to_datetime(breaks)

            # format date
            breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Top-Down':

        # create model with least absolute deviation cost function
        start = timer()
        model = rpt.Binseg(model="l1").fit(series)

        # detect change points
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for change point
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    return breaks, elapsed_time


def company_information(data):

    '''Return company information of selected stock'''

    return data.info['longBusinessSummary']


def selection_mapping_stock_codes(input):

    '''Map company names to their corresponding share codes'''

    mapping = companies.companies

    return mapping[input]


def datetime_to_date_rpt(breaks_rpt):
    '''Format datetime to date'''

    dates = []

    for i in breaks_rpt:
        dates.append(i.date())

    return dates

def date_extraction_of_segments(segments):
    '''
    1. Input segments.
    2. Return start and end date of each segment.
    '''

    dates = []

    for segment in segments:
        start_idx, _ , end_idx, _ = segment
        start_date = custom_strftime(list(df.index)[start_idx].date())
        end_date = custom_strftime(list(df.index)[end_idx].date())
        dates.append((start_date, end_date))

    return dates

def price_change(segments):
    '''
    1. Input segments.
    2. Return price change of each segment.
    '''

    changes = []

    for segment in segments:
        _, start_value , _ , end_value = segment

        abs_change = round(end_value - start_value, 2)
        pct_change = round(((end_value - start_value) / start_value)*100, 2)
        changes.append((abs_change, pct_change))

    return changes

def average_membership_degree(data, segments):
    '''
    1. Input series and segments.
    2. Return average membership degree of fuzzy labels.
    '''

    # get startdate and enddate in custom format of each segment
    dates = date_extraction_of_segments(segments)

    # get slope of each segment and calculate corresponding fuzzy variable
    slopes = preprocessing.calc_slope(segments)
    slope_variables, _ = fuzzy.fuzzy_slope(slopes)

    # get variability of each segment
    variabilities = preprocessing.calc_variability(data, segments)
    variability_variables, _ = fuzzy.fuzzy_variability(variabilities)

    # get slope and variability membership degrees of each segment
    _ , slope_truth_values  = fuzzy.fuzzy_slope(slopes)
    _ , variability_truth_values = fuzzy.fuzzy_variability(variabilities)

    # convert list of lists to list
    slope_truth_values = [item for sublist in slope_truth_values for item in sublist]
    variability_truth_values = [item for sublist in variability_truth_values for item in sublist]

    # calculate average membership degree
    summation = sum(slope_truth_values) + sum(variability_truth_values)
    length = len(slope_truth_values + variability_truth_values)
    avg_md = round(summation/length, 2)

    return avg_md

def summary(data, segments):
    '''
    1. Input series and segments.
    2. Create summary of extended information part using fuzzy logic.
    '''

    # get startdate and enddate in custom format of each segment
    dates = date_extraction_of_segments(segments)

    # get slope of each segment and calculate corresponding fuzzy variable
    slopes = preprocessing.calc_slope(segments)
    slope_variables, _ = fuzzy.fuzzy_slope(slopes)

    # get price changes of each segment
    price_changes = price_change(segments)

    # get variability of each segment
    variabilities = preprocessing.calc_variability(data, segments)
    variability_variables, _ = fuzzy.fuzzy_variability(variabilities)

    output = []

    for i in range(len(segments)):
        output.append('Between ' + dates[i][0] + ' and ' + dates[i][1] + ', ' + 'the closing price ' \
                     + str(slope_variables[i][0]) + ' with a change of ' + str(price_changes[i][0]) +'$' + ' (' + str(price_changes[i][1]) + '%)' \
                     + ' and a ' + str(variability_variables[i][0]) + ' volatility ' + '(' + str(variabilities[i]) + ').')


    return output

# ----------------------------------
# App Logic
# ----------------------------------

# title of app
st.write('''
# Summarising time series data by natural language
This models aims to create a brief summary of stock exchange data. Change point detection algorithms are applied to extract essential information of the time series. This information is then used for generating the linguistic description of the data.''')

# get full company names
full_name_companies = list(companies.companies.keys())

# create selectbox for companies
company = st.selectbox(
     'Select a company of the S&P 500 market index',
     full_name_companies)

tickerDF = ''

# selection process
if company != "Search for a company":

    # map company name to stock code
    stock = selection_mapping_stock_codes(company)

    # get stock data of selected company
    data = yf.Ticker(stock)

    # display company information
    with st.beta_expander("About this company"):
          st.write(company_information(data))

    # identify max and min date of selected stock to restrict date selection
    eval_date = data.history(period = 'max', interval = '1d')
    min_date = min(eval_date.index.date)
    max_date = max(eval_date.index.date)

    end_date = datetime.today() - timedelta(days=1)
    cur_year = datetime.today().year
    string_date = str(cur_year) + '-01-04'

    # transform string date to datetime format
    start_date = datetime.strptime(string_date, '%Y-%m-%d').date()

    # whitespace
    st.write("")

    st.write('''
    Select a time period to narrow down the analysis''')

    # create two columns to display date pickers side-by-side
    col1, col2 = st.beta_columns(2)
    with col1:
        start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    with col2:
        end_date = st.date_input(label='End date', value=end_date, max_value=max_date)

    # check whether selected dates are weekend days
    if start_date.weekday() >= 5 or end_date.weekday() >= 5:
        st.warning('Please select weekdays')
        st.stop()
    if start_date > end_date:
        st.warning('Please choose a start date smaller than the end date')
        st.stop()

    delta = end_date - start_date

    if delta.days >= 7 and delta.days < 60:
        sections = floor(delta.days / 7)
        st.write("Time span:", delta.days)
        st.write("Number of sections:", sections)

    # get stock data filtered by selected date range
    df = data.history(interval = '1d', start = start_date, end = end_date)

    st.write("")

    n_breaks = st.slider('Select the number of change points', 3, 20, 4)
    st.write('You selected ' + str(n_breaks) + ' change points to decompose time series.')

    # list algorithms
    algorithms = ['Bottom-Up', 'Top-Down', 'Window Sliding (40)', 'Window Sliding (60)', 'Window Sliding (100)', 'Window Sliding (200)', 'Dynamic Programming (Optimal Method)']


    with st.form(key='algorithm_selection'):
        algorithm = st.selectbox('Select change point detection algorithm to create linguistic summary of the data course', algorithms) #help='test')
        submit_button = st.form_submit_button(label='Apply')

    # run model only if selection has been made
    if not submit_button:
        st.stop()

    # rename column name for better readability in the user interface
    df.rename(columns={'Close': 'Close Price'}, inplace = True)

    # space
    st.write('')

    # add title to chart
    st.write('''
    ### Closing stock price and approximation by change point detection algorithm''')

    # space
    st.write('')

    # display chart
    chart = st.line_chart(round(df['Close Price'], 4))

    y = np.array(round(df['Close Price'], 4).tolist())
    ts = round(df['Close Price'], 4)

    # detect change points
    with st.spinner('Approximation running...'):
        break_points, elapsed_time = changepoint_detection(algorithm, y)

    # create segments with identified change points
    segments = create_segments(break_points, df)

    cur_index = []
    cur_data = []

    # create visualisation of ground truth and approximation/segmentation
    for i in range(len(segments)):
            cur_index.append(list(df.index)[segments[i][0]])
            cur_data.append(segments[i][1])
            if i == len(segments)-1:
                cur_index.append(list(df.index)[segments[i][0]])
                cur_data.append(segments[i][1])
                cur_index.append(list(df.index)[segments[i][2]])
                cur_data.append(segments[i][3])
            df_test = pd.DataFrame(data = cur_data, index = cur_index, columns = [algorithm + ' Algorithm'])
            chart.add_rows(df_test)
            time.sleep(0.25)

    # calculate approximation error (between ground truth and approximation)
    sse = sumsquared_error(y.tolist(), segments)
    mse = meansquared_error(y.tolist(), segments)

    # calculate average membership degree
    average_md = average_membership_degree(y, segments)

    # join general information and extended information to create final description
    output = summary_price(df) + ' ' + ' '.join(summary(y, segments))

    # display summary
    st.write('''
    ### Textual summary for closing stock price of ''', company)
    with st.beta_expander('Chronological summary'):
        st.write(output)

    # display performance details
    st.write('''
    ### Algorithm details''')
    with st.beta_expander("Performance information of algorithm"):
        st.write('Number of identified change points: ', len(segments), \
        ', Elapsed time (sec): ', elapsed_time, ', Sum of squared error: ', sse, \
        ' Mean of squared error: ', mse, 'Average membership degree: ', average_md)
