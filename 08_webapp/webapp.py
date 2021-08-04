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
import segment
import wrappers
import fit

# algorithms
import jenkspy
import ruptures as rpt
from fastpip import pip

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
    """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""

    total_error = []

    for segment in segments:
        x0,y0,x1,y1 = segment
        p, error = leastsquareslinefit(sequence,(x0,x1))
        total_error.append(round(error, 3))

    sumsquared_error = round(sum(total_error), 2)

    return sumsquared_error

def meansquared_error(sequence, segments):
    """Return the mean of squared errors for a least squares line fit of one segment of a sequence"""

    total_error = []

    for segment in segments:
        x0,y0,x1,y1 = segment
        p, error = leastsquareslinefit(sequence,(x0,x1))
        total_error.append(error)

    meansquared_error = round(mean(total_error), 2)

    return meansquared_error

def suffix(d):
    return 'th' if 11<=d<=13 else {1:'st',2:'nd',3:'rd'}.get(d%10, 'th')

def custom_strftime(t):

    format = '%B {S}, %Y'
    string_date = t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))

    return string_date

def max_price(df):

    max_value = 0
    index = None

    for i, value in enumerate(df['Close Price']):
        if value > max_value:
            max_value = round(value, 2)
            index = df['Close Price'].index[i].date()
            string_date = custom_strftime(index)

    return max_value, string_date

def min_price(df):

    min_value = 999999
    index = 0

    for i, value in enumerate(df['Close Price']):
        if value < min_value:
            min_value = round(value, 2)
            index = df['Close Price'].index[i].date()
            string_date = custom_strftime(index)

    return min_value, string_date

def min_date(df):

    min_date = custom_strftime(df.index.min())

    return min_date

def max_date(df):

    max_date = custom_strftime(df.index.max())

    return max_date

def summary_price(df):

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

    # sentence0 = opening + mindate + glue + maxdate + date_element + str(max_value) + glue_2 + date_1 + glue_3 \
    #             + str(min_value) + glue_4 + date_2 + eos
    sentence0 = opening + str(custom_strftime(df.index.min().date())) + glue + str(custom_strftime(df.index.max().date())) + date_element + str(max_value) + glue_2 + date_1 + glue_3 \
                + str(min_value) + glue_4 + date_2 + eos

    return sentence0


def create_segments(breaks, df):
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

def breakpoint_detection(algorithm, series):

    # Test model
    if algorithm == 'Pelt':

        # fit model
        start = timer()
        model = rpt.Pelt(model="l2").fit(series)

        # detect breakpoints
        breaks_pelt = model.predict(pen=n_breaks)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for breakpoint
        breaks = []
        for i in breaks_pelt:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    # if algorithm == 'Fisher-Jenks':
    #
    #     # detect breakpoints
    #     start = timer()
    #     breaks_pp = jenkspy.jenks_breaks(series, nb_class=n_breaks-1)
    #     end = timer()
    #     elapsed_time = round(end - start, 2)
    #     # get corresponding date for breakpoint
    #     breaks = []
    #     for i in breaks_pp:
    #         idx = ts.index[ts == i]
    #         breaks.append(idx)
    #
    #     breaks = datetime_to_date_jkp(breaks)

    if algorithm == 'Dynamic Programming':

        # train model
        start = timer()
        model = rpt.Dynp(model="l2").fit(series)

        # detect breakpoints
        breaks_pp = model.predict(n_bkps=n_breaks-1)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for breakpoint
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    # if algorithm == 'Perceptually Important Points':
    #
    #     # concatenate values and index
    #     data = []
    #
    #     for i, value in enumerate(series.tolist()):
    #         data.append((i, value))
    #
    #     # detect breakpoints
    #     start = timer()
    #     breaks_pp = pip(data,n_breaks)
    #     end = timer()
    #     elapsed_time = round(end - start, 2)
    #
    #     # get corresponding date for breakpoint
    #     breaks = []
    #
    #     for i in range(len(breaks_pp)):
    #         breaks.append(list(df.index)[breaks_pp[i][0]].date())

    if algorithm == 'Bottom-Up':

        # train model
        start = timer()
        model = rpt.BottomUp(model="l2").fit(series)

        # detect breakpoints
        #breaks_pp = model.predict(n_bkps=n_breaks-1)
        breaks_pp = model.predict(pen=n_breaks)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for breakpoint
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Sliding Window':

        # train model
        start = timer()
        model = rpt.Window(width=40, model="l2").fit(series)

        # detect breakpoints
        #breaks_pp = model.predict(n_bkps=n_breaks-1)
        breaks_pp = model.predict(pen=n_breaks)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for breakpoint
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)

    if algorithm == 'Binary Segmentation':

        # train model
        start = timer()
        model = rpt.Binseg(model="l2").fit(series)

        # detect breakpoints
        #breaks_pp = model.predict(n_bkps=n_breaks-1)
        breaks_pp = model.predict(pen=n_breaks)
        end = timer()
        elapsed_time = round(end - start, 2)

        # get corresponding date for breakpoint
        breaks = []
        for i in breaks_pp:
            breaks.append(ts.index[i-1])
        breaks = pd.to_datetime(breaks)

        # format date
        breaks = datetime_to_date_rpt(breaks)


    return breaks, elapsed_time

def get_iqr_start_end_values(data, segments):

    series = []

    for segment in segments:
        start_idx, _ , end_idx, _ = segment
        series.append(data[start_idx:end_idx])

    range_indexes = []

    for i in series:
        series_sorted = sorted(i)
        index_75th = (len(series_sorted)-1) * 75 / 100.0
        index_75th = series_sorted[int(index_75th)]
        index_25th = (len(series_sorted)-1) * 25 / 100.0
        index_25th = series_sorted[int(index_25th)]
        range_indexes.append((index_25th, index_75th))

    return range_indexes

def get_change_of_iqr_start_end_values(tuples):

    pct_change = []

    for i in tuples:
        pct_change.append(round(((i[1]-i[0]) / i[0])*100, 2))

    return pct_change


def max_error_value(series):

    min_value = math.floor(min(series))
    max_value = math.floor(max(series))
    range_value = max_value - min_value

    error_rate = 0.75 * range_value

    return round(error_rate, 1)

def company_information(data):
    return data.info['longBusinessSummary']

# mapping dict for algorithm selection
def selection_mapping(input):

     mapping = {'Pelt' : 'Pelt',
                'Fisher-Jenks': 'Fisher-Jenks',
                'Dynamic Programming': 'Dynamic Programming',
                'Perceptually Important Points' : 'Pip',
                'Bottom-Up' : 'Bottom-Up',
                'Sliding Window' : 'Sliding Window',
                'Binary Segmentation' : 'Binary Segmentation'}


     return mapping[input]

# mapping dict for stock codes
def selection_mapping_stock_codes(input):

    mapping = companies.companies

    return mapping[input]

def datetime_to_date_jkp(breaks_jkp):

    break_points = []

    for i in breaks_jkp:
        for date in i:
            break_points.append(date.date())

    return break_points

def datetime_to_date_rpt(breaks_rpt):

    dates = []

    for i in breaks_rpt:
        dates.append(i.date())

    return dates

def date_extraction_of_segments(segments):

    dates = []

    for segment in segments:
        start_idx, _ , end_idx, _ = segment
        start_date = custom_strftime(list(df.index)[start_idx].date())
        end_date = custom_strftime(list(df.index)[end_idx].date())
        dates.append((start_date, end_date))

    return dates

def price_change(segments):

    changes = []

    for segment in segments:
        _, start_value , _ , end_value = segment

        abs_change = round(end_value - start_value, 2)
        pct_change = round(((end_value - start_value) / start_value)*100, 2)
        changes.append((abs_change, pct_change))

    return changes

def breakpoint_summary(data, segments):

    # get startdate and enddate in custom format of each segment
    dates = date_extraction_of_segments(segments)

    # get slope of each segment and calculate corresponding fuzzy variable
    slopes = preprocessing.calc_slope(segments)
    slope_variables = fuzzy.fuzzy_slope(slopes)

    # get price changes of each segment
    price_changes = price_change(segments)

    # get variability of each segment
    variabilities = preprocessing.calc_variability(data, segments)
    variability_variables = fuzzy.fuzzy_variability(variabilities)


    output = []

    for i in range(len(segments)):
        output.append('Between ' + dates[i][0] + ' and ' + dates[i][1] + ', ' + 'the closing price ' \
                     + str(slope_variables[i][0]) + ' with a change of ' + str(price_changes[i][0]) +'$' + ' (' + str(price_changes[i][1]) + '%)' \
                     + ' and a ' + str(variability_variables[i][0]) + ' volatility ' + '(' + str(variabilities[i]) + ').')



    return output

st.write('''
# Summarising time series data by natural language
This models aims to create a brief summary of stock exchange data. Approximation algorithms are applied to extract essential information of the time series. This information is then used for generating the linguistic description of the data.''')

full_name_companies = list(companies.companies.keys())

company = st.selectbox(
     'Select a company of the S&P 500 market index',
     full_name_companies)

tickerDF = ''

if company != "Search for a company":

    # map company name to stock code
    stock = selection_mapping_stock_codes(company)

    # get stock data of selected company
    data = yf.Ticker(stock)

    # display company information
    with st.beta_expander("About this company"):
          st.write(company_information(data))

    # identify max and min date of selected stock
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

    # identify max error
    # series = list(round(df.Close, 4))
    # length_of_series = len(series)

    # max_error = max_error_value(series)
    # st.write('Max error rate based on selected time window: ', max_error)

    st.write("")

    #n_breaks = st.slider('Select the number of breakpoints', 0, 20, 2)

    n_breaks = st.slider('Select the number of breakpoints',  50, 1000000, 100, step=20)
    st.write('You selected ' + str(n_breaks) + ' breakpoints to decompose time series.')

    # list algorithms
    #algorithms = ['Perceptually Important Point', 'Bottom-Up', 'Binary Segmentation', 'Sliding Window', 'Fisher-Jenks', 'Dynamic Programming']
    algorithms = ['Pelt', 'Bottom-Up', 'Binary Segmentation', 'Sliding Window', 'Fisher-Jenks', 'Dynamic Programming']


    with st.form(key='algorithm_selection'):
        algorithm = st.selectbox('Select an approximation algorithm to create linguistic summary of the data', algorithms) #help='test')
        submit_button = st.form_submit_button(label='Apply')


    # run model only if selection has been made
    if not submit_button:
        st.stop()

    # rename column name for better readability in the user interface
    df.rename(columns={'Close': 'Close Price'}, inplace = True)

    # display chart
    chart = st.line_chart(round(df['Close Price'], 4))

    #segments = segment_detection(df, max_error, algorithm)
    y = np.array(round(df['Close Price'], 4).tolist())
    ts = round(df['Close Price'], 4)

    # convert breakpoints into date format
    with st.spinner('Approximation running...'):
        break_points, elapsed_time = breakpoint_detection(algorithm, y)

    # create segments with break_points
    segments = create_segments(break_points, df)

    cur_index = []
    cur_data = []

    for i in range(len(segments)):
            cur_index.append(list(df.index)[segments[i][0]])
            cur_data.append(segments[i][1])
            if i == len(segments)-1:
                cur_index.append(list(df.index)[segments[i][0]])
                cur_data.append(segments[i][1])
                cur_index.append(list(df.index)[segments[i][2]])
                cur_data.append(segments[i][3])
            df_test = pd.DataFrame(data = cur_data, index = cur_index, columns = [algorithm + ' Approximation'])
            chart.add_rows(df_test)
            time.sleep(0.25)

    # calculate approximation error
    sse = sumsquared_error(y.tolist(), segments)
    mse = meansquared_error(y.tolist(), segments)

    st.write('''
    ### Algorithm details''')
    with st.beta_expander("Performance information of algorithm"):
        st.write('Number of identified breakpoints: ', len(segments), \
        ', Elapsed time (sec): ', elapsed_time, ', Sum of squared error: ', sse, \
        ' Mean of squared error: ', mse)

    st.write("")

    output = summary_price(df) + ' ' + ' '.join(breakpoint_summary(y, segments))

    st.write('''
    ### Textual Summary''')

    st.write(output)


st.write("")
st.write("")
