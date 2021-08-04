import numpy as np

def calc_variability(data, segments):

    variability_series = []

    for i in range(len(segments)):
        start, _, end, _ = segments[i]
        x = np.array(data[start:end+1])
        #segment_iqr = round(iqr(x), 4)
        segment_sd = np.std(x)
        segment_avg = np.average(x)
        coefficient_of_variance = round((segment_sd / segment_avg), 2)
        variability_series.append(coefficient_of_variance)

    return variability_series

def calc_slope(segments):
    '''
    1. Input segment series.
    2. Return angle of slope.
    '''
    slope_series = []
    for segment in segments:
        x0, y0, x1, y1 = segment
        angle = round((np.rad2deg(np.arctan2(y1 - y0, x1 - x0))), 2)
        slope_series.append(angle)

    return slope_series
