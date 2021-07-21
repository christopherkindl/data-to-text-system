import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import companies
import time
#from datetime import date as dt
from datetime import datetime, timedelta
import segment
import wrappers
from test_list import list_segments # import test segments
import fit
import math
from math import floor
from fuzzylogic.functions import R, S, alpha, triangular, bounded_linear, trapezoid
from fuzzylogic.classes import Domain
from fuzzylogic.hedges import plus, minus, very
import jenkspy
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
#from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
#from matplotlib.lines import Line2D

# ----------------------------------
# Fuzzylogic
# ----------------------------------

# slope = Domain("slope", -90, 90, res=0.01)
# slope.quickly_decreasing = S(-90+11.25, -90+32.5)
# slope.decreasing = triangular(-77.5, -35)
# slope.slowly_decreasing = triangular(-55, -12.5)
# slope.constant = triangular(-32.5, 32.5)
# slope.slowly_increasing = triangular(12.5, 55)
# slope.increasing = triangular(35, 77.5)
# slope.quickly_increasing = R(90-32.5, 90-11.25)
#
# def fuzzy_slope(series):
#
#     total_sum = {'quickly decreasing' : [],
#                  'decreasing' : [],
#                  'slowly decreasing' : [],
#                  'constant' : [],
#                  'slowly increasing' : [],
#                  'increasing' : [],
#                  'quickly increasing' : []}
#
#     indexes = {'quickly decreasing' : [],
#                'decreasing' : [],
#                'slowly decreasing' : [],
#                'constant' : [],
#                'slowly increasing' : [],
#                'increasing' : [],
#                'quickly increasing' : []}
#
#     for index, i in enumerate(series):
#         total_sum['quickly decreasing'].append(round(slope.quickly_decreasing(i), 2))
#         if round(slope.quickly_decreasing(i), 2) > 0:
#             indexes['quickly decreasing'].append(index)
#
#         total_sum['decreasing'].append(round(slope.decreasing(i), 2))
#         if round(slope.decreasing(i), 2) > 0:
#             indexes['decreasing'].append(index)
#
#         total_sum['slowly decreasing'].append(round(slope.slowly_decreasing(i), 2))
#         if round(slope.slowly_decreasing(i), 2) > 0:
#             indexes['slowly decreasing'].append(index)
#
#         total_sum['constant'].append(round(slope.constant(i), 2))
#         if round(slope.constant(i), 2) > 0:
#             indexes['constant'].append(index)
#
#         total_sum['slowly increasing'].append(round(slope.slowly_increasing(i), 2))
#         if round(slope.slowly_increasing(i), 2) > 0:
#             indexes['slowly increasing'].append(index)
#
#         total_sum['increasing'].append(round(slope.increasing(i), 2))
#         if round(slope.increasing(i), 2) > 0:
#             indexes['increasing'].append(index)
#
#         total_sum['quickly increasing'].append(round(slope.quickly_increasing(i), 2))
#         if round(slope.quickly_increasing(i), 2) > 0:
#             indexes['quickly increasing'].append(index)
#
#     number_of_segments = len(series)
#
#     # calculate truth value
#     n = 0
#
#     for i in total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))):
#         if i != 0:
#             n += i
#
#     # calculate quantifier
#     q = 0
#
#     for i in total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))):
#         if i != 0:
#             q += 1
#
#
#
#     value = round((q / len(series)), 2)
#
#     quantifier_value = quantifier(value)
#
#     return max(total_sum, key = lambda x: sum(total_sum.get(x))), round(sum(total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x))))), 2), round((n/number_of_segments), 2), indexes.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))), quantifier_value
#
# duration = Domain("duration", 0, 180, res=0.01)
# duration.very_short = S(0+7, 0+14)
# duration.short = trapezoid(7, 14, 35, 42)
# duration.medium = trapezoid(35, 42, 42+21, 42+28)
# duration.long = trapezoid(63, 70, 70+21, 70+28)
# duration.very_long = R(91, 98)
#
# def fuzzy_duration(series):
#
#     total_sum = {'very short' : [],
#                  'short' : [],
#                  'medium' : [],
#                  'long' : [],
#                  'very long' : []}
#
#     for i in series:
#         total_sum['very short'].append(round(duration.very_short(i), 2))
#         total_sum['short'].append(round(duration.short(i), 2))
#         total_sum['medium'].append(round(duration.medium(i), 2))
#         total_sum['long'].append(round(duration.long(i), 2))
#         total_sum['very long'].append(round(duration.very_long(i), 2))
#
#
#     number_of_segments = len(series)
#
#     n = 0
#
#     for i in total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))):
#         if i != 0:
#             n += i
#
#     return max(total_sum, key = lambda x: sum(total_sum.get(x))), sum(total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))))
#
#
# variability = Domain("variability", 0, 1, res=0.01)
# variability.very_high = R(0.75, 4/5)
# variability.high = trapezoid(0.55, 3/5, 0.75, 4/5)
# variability.medium = trapezoid(0.35, 2/5, 0.55, 3/5)
# variability.low = trapezoid(0.15, 1/5, 0.35, 2/5)
# variability.very_low = S(0.15, 1/5)
#
# def fuzzy_variability(series):
#
#     total_sum = {'very high' : [],
#                  'high' : [],
#                  'medium' : [],
#                  'low' : [],
#                  'very low' : []}
#
#
#     for i in series:
#         total_sum['very high'].append(round(variability.very_high(i), 2))
#         total_sum['high'].append(round(variability.high(i), 2))
#         total_sum['medium'].append(round(variability.medium(i), 2))
#         total_sum['low'].append(round(variability.low(i), 2))
#         total_sum['very low'].append(round(variability.very_low(i), 2))
#
#
#     number_of_segments = len(series)
#
#     n = 0
#
#     for i in total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))):
#         if i != 0:
#             n += i
#
#     return max(total_sum, key = lambda x: sum(total_sum.get(x))), sum(total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))))
#
# # ----------------------------------
# # preprocessing
# # ----------------------------------
#
# def quantifier(value):
#
#     quantifier = {
#     'almost all of ' : 0,
#     'most of ' : 0,
#     'at least a half of ' : 0,
#     'at least a third of ' : 0,
#     'some of ' : 0,
#     'none of ' : 0
#     }
#
#     if value >= 0.85:
#         quantifier['almost all of '] += 1
#     if value >= 0.7:
#         quantifier['most of '] += 1
#     if value >= 0.5:
#         quantifier['at least a half of '] += 1
#     if value >= 0.3:
#         quantifier['at least a third of '] += 1
#     if value > 0:
#         quantifier['some of '] += 1
#     if value == 0:
#         quantifier['none of '] += 1
#
#     quantifier_value = max(quantifier, key = quantifier.get)
#
#     return quantifier_value

# ----------------------------------
# Eval Functions
# ----------------------------------

# def calc_duration(segments, filtering):
#     '''
#     1. Input segment series.
#     2. Return segment-specific duration.
#     '''
#     duration_series = []
#     for i in range(len(segments)):
#         if i in filtering:
#             start, _, end, _ = segments[i]
#             length = end-start
#             duration_series.append(length)
#
#     return duration_series
#
# def calc_slope(segments):
#     '''
#     1. Input segment series.
#     2. Return angle of slope.
#     '''
#     slope_series = []
#     for segment in segments:
#         x0, y0, x1, y1 = segment
#         angle = round((np.rad2deg(np.arctan2(y1 - y0, x1 - x0))), 2)
#         slope_series.append(angle)
#
#     return slope_series
#
# def calc_variability(data, segments, filtering):
#     '''
#     1. Input value range as array.
#     2. Return interquartile range.
#     '''
#     variability_series = []
#
#     for i in range(len(segments)):
#         if i in filtering:
#             start, _, end, _ = segments[i]
#             x = np.array(data[start:end+1])
#             #segment_iqr = round(iqr(x), 4)
#             segment_sd = np.std(x)
#             segment_avg = np.average(x)
#             coefficient_of_variance = round((segment_sd / segment_avg), 2)
#             variability_series.append(coefficient_of_variance)
#
#     return variability_series
#
# def calc_slope_L2(segments, filtering):
#     '''
#     1. Input segment series.
#     2. Return angle of slope.
#     '''
#     slope_series = []
#     for i in range(len(segments)):
#         if i not in filtering:
#             x0, y0, x1, y1 = segments[i]
#             angle = round((np.rad2deg(np.arctan2(y1 - y0, x1 - x0))), 2)
#             slope_series.append(angle)
#
#     return slope_series
#
# def calc_duration_L2(segments, filtering):
#     '''
#     1. Input segment series.
#     2. Return segment-specific duration.
#     '''
#     duration_series = []
#     for i in range(len(segments)):
#         if i not in filtering:
#             start, _, end, _ = segments[i]
#             length = end-start
#             duration_series.append(length)
#
#     return duration_series
#
# def calc_variability_L2(data, segments, filtering):
#     '''
#     1. Input value range as array.
#     2. Return interquartile range.
#     '''
#     variability_series = []
#
#     for i in range(len(segments)):
#         if i not in filtering:
#             start, _, end, _ = segments[i]
#             x = np.array(data[start:end+1])
#             #segment_iqr = round(iqr(x), 4)
#             segment_sd = np.std(x)
#             segment_avg = np.average(x)
#             coefficient_of_variance = round((segment_sd / segment_avg), 2)
#             variability_series.append(coefficient_of_variance)
#
#     return variability_series

# ----------------------------------
# Functions
# ----------------------------------

def dynamics(series):
    '''
    1. Input segment series.
    2. Return angle of slope.
    '''
    slope_series = []
    for segment in series:
        x0, y0, x1, y1 = segment
        angle = round((np.rad2deg(np.arctan2(y1 - y0, x1 - x0))), 2)
        slope_series.append(angle)

    return slope_series

def suffix(d):
    return 'th' if 11<=d<=13 else {1:'st',2:'nd',3:'rd'}.get(d%10, 'th')

def custom_strftime(t):

    format = '%B {S} %Y'
    string_date = t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))

    # check format type
    #return t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))
    #string_date = custom_strftime('%B {S} %Y', datetime.now())

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
    sentence0 = opening + str(df.index.min().date()) + glue + str(df.index.max().date()) + date_element + str(max_value) + glue_2 + date_1 + glue_3 \
                + str(min_value) + glue_4 + date_2 + eos

    return sentence0

# def length_of_df(df):
#
#     length = len(df)
#
#     if length >= 7 and length < 60:
#         segment_range = 7
#     if length >= 60 and length < 180:
#         segment_range = 30
#     if length >= 180 and length < 365:
#         segment_range = 90
#     if length >= 365:
#         segment_range = 365
#
#     return segment_range
#
# def segment_divider(tuple_series, segment_range):
#     day_counter = segment_range
#
#     sublist = [[]]
#
#     for date, i in tuple_series:
#         sublist[-1].append((date, i))
#         day_counter -= 1
#         if day_counter == 0:
#             sublist.append([])
#             day_counter = segment_range
#
#     return sublist
#
# def segment_divider(series, segment_range):
#     day_counter = segment_range
#
#     sublist = [[]]
#
#     for i in series:
#         sublist[-1].append(i)
#         day_counter -= 1
#         if day_counter == 0:
#             sublist.append([])
#             day_counter = segment_range
#
#     return sublist
#
# def columns_to_tuples(column1, column2):
#     list_of_tuples = list(zip(column1, round(column2, 2)))
#     return list_of_tuples

# def segment_detection(df, max_error, algorithm):
#
#     # identify index integer based on selected start and end date
#     start_date = list(df.index)[0]
#     start = df.index.get_loc(start_date)
#     end_date = list(df.index)[-1]
#     end = df.index.get_loc(end_date)
#
#     segment_range = length_of_df(df)
#     tuple_series = columns_to_tuples(df.index.date, df.Close)
#     sublist = segment_divider(tuple_series, segment_range)
#
#     # transform input date from datetime to string date
#     #start_date = start_date.strftime('%Y-%m-%d')
#     #end_date = end_date.strftime('%Y-%m-%d')
#     #start_date
#     #end_date
#
#     # get index integer of date index
#     #start = df.index.get_loc(start_date)
#     #end = df.index.get_loc(end_date)
#
#     # create list out of df column
#     series = list(round(df['Close Price'], 2))

    # ------ TEST

def create_segments(breaks_jkp, df):
    segments = []
    start = df.index[0]
    index = list(df.index)
    for breakpoint in breaks_jkp:
        segment = (index.index(start), round(df["Open"][start], 4), index.index(breakpoint), round(df["Open"][breakpoint], 4))
        segments.append(segment)
        start = breakpoint
    if df.index[-1] != breaks_jkp[-1]:
        final = df.index[-1]
        segment = (index.index(breaks_jkp[-1]), round(df["Open"][breaks_jkp[-1]], 4), index.index(final), round(df["Open"][final], 4))
        segments.append(segment)
    return segments

def segment_detection(df, max_error, algorithm):

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
    series = list(round(df['Close Price'], 2))

    #tuple_series = columns_to_tuples(df.index.date, df.Close)

    # detect segments and filter series based on selected time range

    if algorithm == 'bottomupsegment':
        # get segments
        # segments = []
        #
        # for subsegment in list_of_lists:
        #     #print(series)
        #     segments.append(segment.bottomupsegment(subsegment, fit.interpolate, fit.sumsquared_error, max_error))

        segments = segment.bottomupsegment(series, fit.interpolate, fit.sumsquared_error, max_error)

    if algorithm == 'topdownsegment':

        # segments = []
        #
        # for subsegment in list_of_lists:
        #     #print(series)
        #     segments.append(segment.topdownsegment(subsegment, fit.interpolate, fit.sumsquared_error, max_error))

        segments = segment.topdownsegment(series[start:end], fit.interpolate, fit.sumsquared_error, max_error)

    if algorithm == 'slidingwindowsegment':

        # segments = []
        #
        # for subsegment in list_of_lists:
        #     #print(series)
        #     segments.append(segment.slidingwindowsegment(subsegment, fit.interpolate, fit.sumsquared_error, max_error))

        segments = segment.slidingwindowsegment(series[start:end], fit.interpolate, fit.sumsquared_error, max_error)

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

    error_rate = 0.75 * range_value

    return round(error_rate, 1)

def company_information(data):
    return data.info['longBusinessSummary']

# mapping dict for algorithm selection
def selection_mapping(input):

     mapping = {'Bottom-Up':'bottomupsegment',
                'Top-Down':'topdownsegment',
                'Sliding Window': 'slidingwindowsegment'}

     return mapping[input]

def datetime_to_date(breaks_jkp):

    break_points = []

    for i in breaks_jkp:
        for date in i:
            break_points.append(date.date())

    return break_points

# ----------------------------------
# Function for new NLG
# ----------------------------------
#
# def subsegment_calculation(data, segments):
#     """
#     1. Data = see output above
#     """
#
#     # ---------------
#     # LEVEL 1
#     # ---------------
#
#     # calculate slope
#     subsegment_calculations_slope = []
#
#     for subsegment in segments:
#         subsegment_calculations_slope.append(calc_slope(subsegment))
#
#     subsegment_variables_slope = []
#     filter_list = []
#
#     for subsegment in subsegment_calculations_slope:
#         subsegment_variables_slope.append(fuzzy_slope(subsegment))
#         filter_list.append(fuzzy_slope(subsegment)[3])
#
#
#     # calculate duration
#     subsegment_calculations_duration = []
#
#     for i in range(len(segments)):
#         subsegment_calculations_duration.append(calc_duration(segments[i], filter_list[i]))
#
#     subsegment_variables_duration = []
#
#     for subsegment in subsegment_calculations_duration:
#         subsegment_variables_duration.append(fuzzy_duration(subsegment))
#
#
#     # calculate variability
#     subsegment_calculations_variability = []
#
#     for i in range(len(data)):
#         subsegment_calculations_variability.append(calc_variability(data[i], segments[i], filter_list[i]))
#
#     subsegment_variables_variability = []
#
#     for subsegment in subsegment_calculations_variability:
#         subsegment_variables_variability.append(fuzzy_variability(subsegment))
#
#
#     # ---------------
#     # LEVEL 2
#     # ---------------
#
#
#     # calculate slope
#
#     subsegment_calculations_slope_L2 = []
#
#     for i in range(len(segments)):
#         subsegment_calculations_slope_L2.append(calc_slope_L2(segments[i], filter_list[i]))
#
#     subsegment_variables_slope_L2 = []
#
#     for i in range(len(segments)):
#     #for subsegment in subsegment_calculations_slope_L2:
#         if len(subsegment_calculations_slope_L2[i]) != 0:
#             #print(subsegment)
#             subsegment_variables_slope_L2.append(fuzzy_slope(subsegment_calculations_slope_L2[i]))
#             filter_list[i] += fuzzy_slope(subsegment_calculations_slope_L2[i])[3]
#
#
#
#     # calculate duration
#
#     subsegment_calculations_duration_L2 = []
#
#     for i in range(len(segments)):
#         subsegment_calculations_duration_L2.append(calc_duration_L2(segments[i], filter_list[i]))
#
#     subsegment_variables_duration_L2 = []
#
#     for subsegment in subsegment_calculations_slope_L2:
#         if len(subsegment) != 0:
#             subsegment_variables_duration_L2.append(fuzzy_duration(subsegment))
#
#
#     # calculate variability
#     subsegment_calculations_variability_L2 = []
#
#     for i in range(len(data)):
#         subsegment_calculations_variability_L2.append(calc_variability_L2(data[i], segments[i], filter_list[i]))
#
#     subsegment_variables_variability_L2 = []
#
#     for subsegment in subsegment_calculations_variability_L2:
#         if len(subsegment) != 0:
#             subsegment_variables_variability_L2.append(fuzzy_variability(subsegment))
#
#
#     # evaluate quantifier
#     quantifiers = []
#     values = []
#
#     for i in range(len(subsegment_calculations_slope_L2)):
#         values.append(round((len(subsegment_variables_slope_L2[i][3])/len(subsegment_calculations_slope[i])), 2))
#         quantifiers.append(quantifier(round((len(subsegment_variables_slope_L2[i][3])/len(subsegment_calculations_slope[i])), 2)))
#
#
#     # ---------------
#     # LEVEL 3
#     # ---------------
#
#     # calculate slope
#
#
#     subsegment_calculations_slope_L3 = []
#
#     for i in range(len(segments)):
#         subsegment_calculations_slope_L3.append(calc_slope_L2(segments[i], filter_list[i]))
#
#     subsegment_variables_slope_L3 = []
#
#     for i in range(len(segments)):
#     #for subsegment in subsegment_calculations_slope_L2:
#         if len(subsegment_calculations_slope_L3[i]) != 0:
#             #print(subsegment)
#             subsegment_variables_slope_L3.append(fuzzy_slope(subsegment_calculations_slope_L3[i]))
#             #filter_list[i].append(fuzzy_slope(subsegment_calculations_slope_L3[i])[3])
#
#     # calculate duration
#
#     subsegment_calculations_duration_L3 = []
#
#     for i in range(len(segments)):
#         subsegment_calculations_duration_L3.append(calc_duration_L2(segments[i], filter_list[i]))
#
#     subsegment_variables_duration_L3 = []
#
#     for subsegment in subsegment_calculations_slope_L3:
#         if len(subsegment) != 0:
#             subsegment_variables_duration_L3.append(fuzzy_duration(subsegment))
#
#
#     # calculate variability
#     subsegment_calculations_variability_L3 = []
#
#     for i in range(len(data)):
#         subsegment_calculations_variability_L3.append(calc_variability_L2(data[i], segments[i], filter_list[i]))
#
#     subsegment_variables_variability_L3 = []
#
#     for subsegment in subsegment_calculations_variability_L3:
#         if len(subsegment) != 0:
#             subsegment_variables_variability_L3.append(fuzzy_variability(subsegment))
#
#     # evaluate quantifier
#     quantifiers_L3 = []
#     values_L3 = []
#
#     for i in range(len(subsegment_calculations_slope_L3)):
#         values_L3.append(round((len(subsegment_variables_slope_L3[i][3])/len(subsegment_calculations_slope[i])), 2))
#         quantifiers_L3.append(quantifier(round((len(subsegment_variables_slope_L3[i][3])/len(subsegment_calculations_slope[i])), 2)))
#
#
#
#     return subsegment_variables_slope, subsegment_variables_duration, subsegment_variables_variability, subsegment_variables_slope_L2, subsegment_variables_duration_L2, subsegment_variables_variability_L2, quantifiers, subsegment_variables_slope_L3, subsegment_variables_duration_L3, subsegment_variables_variability_L3, quantifiers_L3
#
#
# def length_of_df(df):
#
#     length = len(df)
#
#     if length >= 7 and length < 60:
#         segment_range = 7
#     if length >= 60 and length < 180:
#         segment_range = 30
#     if length >= 180 and length < 365:
#         segment_range = 90
#     if length >= 365:
#         segment_range = 365
#
#     return segment_range
#
# def columns_to_tuples(column1, column2):
#     tuple_series = list(zip(column1, round(column2, 2)))
#     return tuple_series
#
# def segment_divider_tuples(tuple_series, segment_range):
#     day_counter = segment_range
#
#     tuple_sublist = [[]]
#
#     for date, i in tuple_series:
#         tuple_sublist[-1].append((date, i))
#         day_counter -= 1
#         if day_counter == 0:
#             tuple_sublist.append([])
#             day_counter = segment_range
#
#     return tuple_sublist
#
# def list_transformation(tuple_series):
#
#     # retrieve only numeric values of tuple series
#     series = []
#
#     for lis in tuple_series:
#         series.append([])
#         for value in lis:
#             series[-1].append(value[-1])
#
#     return series
#
#
#
# def flat_list(segments):
#
#     flat_list = [item for sublist in segments for item in sublist]
#
#     return flat_list
#
#
# def flatten_list(segments):
#     flat_list = []
#     curr_idx = 0
#     count = 0
#     for sublist in segments:
#         for segment in sublist:
#             start, val_start, end, val_end = segment
#             flat_list.append((curr_idx, val_start, curr_idx+(end-start), val_end))
#             curr_idx += (end-start)
#         curr_idx += 1
#     return flat_list
#
# def date_assignment(tuple_series):
#
#     dates = []
#
#     for i in range(len(tuple_series)):
#         dates.append([tuple_series[i][0][0], tuple_series[i][-1][0]])
#
#     return dates
#
# def merge_sentences(sentence_list, sentence_dict_1, sentence_dict_2):
#     final_output = []
#
#     for i in sentence_dict_1.keys():
#         #sentence_list[i] = sentence_list[i] + ' ' + sentence_dict_1[i] + '.' + ' ' + sentence_dict_2[i] + '.'
#         final_output.append(sentence_list[i] + ' ' + sentence_dict_1[i] + ' ' + sentence_dict_2[i] + '.')
#     return final_output
#
# def nlg(tuple_series, subsegment_calculations):
#
#     # -----------------
#     # Level 1
#     # -----------------
#
#     temp_sentences = []
#
#     for dates in date_assignment(tuple_series):
#          # add customdate format
#          temp_sentences.append('between ' + str(dates[0]) + ' and ' + str(dates[1]))
#
#     sentences = []
#
#     for i in range(len(temp_sentences)):
#         sentences.append(temp_sentences[i] + ', ' + subsegment_calculations[0][i][4] + 'trends are ' + subsegment_calculations[0][i][0] + '.')
#
#     extended_sentences = []
#
#     for i in range(len(temp_sentences)):
#         extended_sentences.append((subsegment_calculations[0][i][0] + ' trends tend to be ' + subsegment_calculations[1][i][0] + ' and of a ' + subsegment_calculations[2][i][0] + ' variability'))
#
#     final_sentences = []
#
#     for i in range(len(temp_sentences)):
#         final_sentences.append(sentences[i] + ' ' + extended_sentences[i] + ' during that period.')
#
#
#     # -----------------
#     # Level 2
#     # -----------------
#
#     temp_sentences_L2 = {}
#     extended_sentences_L2 = {}
#
#     if len(subsegment_calculations[3]) != 0:
#
#         for i in range(len(subsegment_calculations[3])):
#             if len(subsegment_calculations[3][i]) != 0:
#                 temp_sentences_L2.update({i:subsegment_calculations[6][i] + 'trends are ' + subsegment_calculations[3][i][0] + '.'})
#
#         for i in range(len(subsegment_calculations[4])):
#             if len(subsegment_calculations[4][i]) != 0:
#                 extended_sentences_L2.update({i:subsegment_calculations[3][i][0] + ' trends tend to be ' + subsegment_calculations[4][i][0] + ' and of a ' + subsegment_calculations[5][i][0] + ' variability'})
#
#     #final_output = merge_sentences(final_sentences, temp_sentences_L2)
#
#     # -----------------
#     # Level 3
#     # -----------------
#
#     temp_sentences_L3 = {}
#     extended_sentences_L3 = {}
#
#     if len(subsegment_calculations[7]) != 0:
#
#         for i in range(len(subsegment_calculations[7])):
#             if len(subsegment_calculations[7][i]) != 0:
#                 temp_sentences_L3.update({i:subsegment_calculations[10][i] + 'trends are ' + subsegment_calculations[7][i][0] + '.'})
#
#         for i in range(len(subsegment_calculations[8])):
#             if len(subsegment_calculations[8][i]) != 0:
#                 extended_sentences_L3.update({i:subsegment_calculations[7][i][0] + ' trends tend to be ' + subsegment_calculations[8][i][0] + ' and of a ' + subsegment_calculations[9][i][0] + ' variability'})
#
#     final_output = merge_sentences(final_sentences, temp_sentences_L2, extended_sentences_L2)
#
#     return final_output

# ----------------------------------
# App Logic
# ----------------------------------

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

tickerDF = ''

if stock != "Search for company share code":

    # get stock data of selected company
    data = yf.Ticker(stock)

    # display company information
    # with st.beta_expander("About this company"):
    #      st.write(company_information(data))

    # identify max and min date of selected stock
    eval_date = data.history(period = 'max', interval = '1d')
    min_date = min(eval_date.index.date)
    max_date = max(eval_date.index.date)

    # TEST TEST ----

    # df = data.history(period = 'max', interval = '1d')

    # series = list(round(df['Close'], 2))

    # ------

    # initialise date range with year-to-date


    end_date = datetime.today() - timedelta(days=1)
    cur_year = datetime.today().year
    string_date = str(cur_year) + '-01-04'

    # end_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    # start_date = datetime.strptime(max_date, '%Y-%m-%d').date()

    # transform string date to datetime format
    start_date = datetime.strptime(string_date, '%Y-%m-%d').date()

    st.write('''
    Select a time period to narrow down the analysis''')

    # create two columns to display date pickers side-by-side
    col1, col2 = st.beta_columns(2)
    with col1:
        start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    with col2:
        end_date = st.date_input(label='End date', value=end_date, max_value=max_date)



    # user field for date selection
    # start_date = st.date_input(label='Start date', value=start_date, min_value=min_date)
    # end_date = st.date_input(label='End date', value=end_date, max_value=max_date)

    # VALIDATION
    # weekday1 = start_date.weekday()
    # weekday1
    # weekday2 = end_date.weekday()
    # weekday2



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
    series = list(round(df.Close, 4))
    length_of_series = len(series)


    max_error = max_error_value(series)
    st.write('Max error rate based on selected time window: ', max_error)

    st.write("")

    n_breaks = st.slider('Select the number of breakpoints', 0, 20, 2)

    col1, col2 = st.beta_columns([.5,1])
    with st.form(key='algorithm_selection'):
    	input = st.radio('Select an approximation algorithm', ['Bottom-Up', 'Top-Down', 'Sliding Window'])
    	submit_button = st.form_submit_button(label='Apply')

    # run model only if selection has been made
    if not submit_button:
        st.stop()

    st.write('You selected a ' + input + ' Approximatation')

    # rename column name for better readability in the user interface
    df.rename(columns={'Close': 'Close Price'}, inplace = True)

    # display chart
    chart = st.line_chart(round(df['Close Price'], 4))

    # time series decomposition

    # assign DF
    # df = tickerDF

    # start_date = list(df.index)[0].date()
    # end_date = list(df.index)[-1].date()

    # use selected algorithm by user
    algorithm = selection_mapping(input)

    # run preprocessing and assign segments
    # segment_range = length_of_df(df)
    # tuple_series = columns_to_tuples(df.index.date, df['Close Price'])
    # tuple_series_pp = segment_divider_tuples(tuple_series, segment_range)
    # list_of_lists = list_transformation(tuple_series_pp)
    # segments_pp = segment_detection(list_of_lists, algorithm)
    # segments_viz =

    segments = segment_detection(df, max_error, algorithm)
    y = np.array(round(df['Close Price'], 4).tolist())
    ts = round(df['Close Price'], 4)


    breaks = jenkspy.jenks_breaks(y, nb_class=n_breaks-1)

    breaks_jkp = []
    for v in breaks:
        idx = ts.index[ts == v]
        breaks_jkp.append(idx)


    # convert breakpoints into date format
    break_points = datetime_to_date(breaks_jkp)

    # create segments with break_points

    segments = create_segments(break_points, df)

    cur_index = []
    cur_data = []

    # segments = flatten_list(segments_pp)
    # st.write('Number of identified segments: ', len(segments))
    # identified segments
    # st.write('Segments identified: ', segments_identified)

    for i in range(len(segments)):
            cur_index.append(list(df.index)[segments[i][0]])
            cur_data.append(segments[i][1])
            if i == len(segments)-1:
                cur_index.append(list(df.index)[segments[i][0]])
                cur_data.append(segments[i][1])
                cur_index.append(list(df.index)[segments[i][2]])
                cur_data.append(segments[i][3])
            df_test = pd.DataFrame(data = cur_data, index = cur_index, columns = [input + ' Approximation'])
            chart.add_rows(df_test)
            # identified segments
            # st.write('Segments identified: ', len(cur_data))
            time.sleep(0.1)

    st.write('Number of identified segments: ', len(segments))


    # ----------------
    # Pyplot Approach
    # ----------------

    # plt.figure()
    # plt.plot(ts, label='data')
    # plt.title('Stock price of Tesla')
    # print_legend = True
    # for i in breaks_jkp:
    #     if print_legend:
    #         plt.axvline(i, color='red',linestyle='dashed', label='breaks')
    #         print_legend = False
    #     else:
    #         plt.axvline(i, color='red',linestyle='dashed')
    #
    #
    # st.pyplot(plt)

    # subsegment_calculations = subsegment_calculation(list_of_lists, segments_pp)
    # final_output = nlg(tuple_series_pp, subsegment_calculations)

    # cur_index = []
    # cur_data = []
    #
    # # segments = flatten_list(segments_pp)
    # # st.write('Number of identified segments: ', len(segments))
    # # identified segments
    # # st.write('Segments identified: ', segments_identified)
    #
    # for i in range(len(segments)):
    #         cur_index.append(list(df.index)[segments[i][0]])
    #         cur_data.append(segments[i][1])
    #         if i == len(segments)-1:
    #             cur_index.append(list(df.index)[segments[i][0]])
    #             cur_data.append(segments[i][1])
    #             cur_index.append(list(df.index)[segments[i][2]])
    #             cur_data.append(segments[i][3])
    #         df_test = pd.DataFrame(data = cur_data, index = cur_index, columns = [input + ' Approximation'])
    #         chart.add_rows(df_test)
    #         # identified segments
    #         # st.write('Segments identified: ', len(cur_data))
    #         time.sleep(0.1)


    # cur_index = []
    # cur_data = []
    #
    # # segments = flatten_list(segments_pp)
    # # st.write('Number of identified segments: ', len(segments))
    # # identified segments
    # # st.write('Segments identified: ', segments_identified)
    #
    # for i in range(len(segments)):
    #         cur_index.append(list(df.index)[segments[i][0]])
    #         cur_data.append(segments[i][1])
    #         if i == len(segments)-1:
    #             cur_index.append(list(df.index)[segments[i][0]])
    #             cur_data.append(segments[i][1])
    #             cur_index.append(list(df.index)[segments[i][2]])
    #             cur_data.append(segments[i][3])
    #         df_test = pd.DataFrame(data = cur_data, index = cur_index, columns = [input + ' Approximation'])
    #         chart.add_rows(df_test)
    #         # identified segments
    #         # st.write('Segments identified: ', len(cur_data))
    #         time.sleep(0.1)
    #
    # st.write('Number of identified segments: ', len(segments))



    # cur_data = []
    # cur_index = []
    #
    # for i in range(len(tickerDF.Open)):
    #     cur_index.append(tickerDF.Open.index[i])
    #     cur_data.append(tickerDF.Open[i])
    #     df = pd.DataFrame(data=cur_data, index = cur_index)
    #     chart.add_rows(df)
    #     time.sleep(0.0025)

    # if tickerDF == '':
    #     st.stop()

    st.write("")

    with st.beta_expander("Summary"):
         st.write(summary_price(df))



#with st.beta_expander("Expand to read"):


# st.write("")
#
# st.write('''
# ### Under the hood
# Algorithms used to identify trends and segments''')
#
# with st.beta_expander("Algorithm 1: Piece-wise Aggregate Approximation"):
#      st.write("""
#          Explaning how the piecewise linear representation is done.
#      """)
#
# with st.beta_expander("Algorithm 2: Discrete Fourier Transform"):
#      st.write("""
#          Explaning how the piecewise linear representation is done.
#      """)
#
# with st.beta_expander("Algorithm 3: Multiple Coefficient Binning"):
#      st.write("""
#          Explaning how the piecewise linear representation is done.
#      """)
#
# with st.beta_expander("Algorithm 4: Symbolic Aggregate approximation"):
#      st.write("""
#          Explaning how the piecewise linear representation is done.
#      """)

# ----------------------------------
# User input: time period
# ----------------------------------


st.write("")
st.write("")
