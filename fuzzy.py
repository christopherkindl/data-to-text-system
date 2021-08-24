from fuzzylogic.functions import R, S, alpha, triangular, bounded_linear, trapezoid
from fuzzylogic.classes import Domain
from fuzzylogic.hedges import plus, minus, very


# define membership functions for trend direction
slope = Domain("slope", -90, 90, res=0.01)
slope.decreased_very_quickly = S(-90+30, -90+35)
slope.decreased_quickly = trapezoid(-60, -55, -40, -35)
slope.decreased = trapezoid(-40, -35, -15, -10)
slope.decreased_slowly = trapezoid(-15, -10, -5, 0)
slope.stable = trapezoid(-2.5, -1, 1, 2.5)
slope.increased_slowly = trapezoid(0, 5, 10, 15)
slope.increased = trapezoid(10, 15, 35, 40)
slope.increased_quickly = trapezoid(35, 40, 55, 60)
slope.increased_very_quickly = R(90-35, 90-30)

# define membership functions for variability
variability = Domain("variability", 0, 1, res=0.01)
variability.very_high = R(0.75, 4/5)
variability.high = trapezoid(3/5, 0.65, 0.75, 4/5)
variability.medium = trapezoid(0.35, 2/5, 3/5, 0.65)
variability.low = trapezoid(1/5, 0.25, 0.35, 2/5)
variability.very_low = S(1/5, 0.25)

def fuzzy_slope(series):
    '''Return linguistic variable for trend direction based on max value of membership function'''

    variables = []
    t_values = []

    for i in series:

        total_sum = {'decreased very quickly' : [],
             'decreased quickly' : [],
             'decreased' : [],
             'decreased slowly' : [],
             'remained stable' : [],
             'increased slowly' : [],
             'increased' : [],
             'increased quickly' : [],
             'increased very quickly' : []}

        # change wording of membership functions
        total_sum['decreased very quickly'].append(round(slope.decreased_very_quickly(i), 2))

        total_sum['decreased quickly'].append(round(slope.decreased_quickly(i), 2))

        total_sum['decreased'].append(round(slope.decreased(i), 2))

        total_sum['decreased slowly'].append(round(slope.decreased_slowly(i), 2))

        total_sum['remained stable'].append(round(slope.stable(i), 2))

        total_sum['increased slowly'].append(round(slope.increased_slowly(i), 2))

        total_sum['increased'].append(round(slope.increased(i), 2))

        total_sum['increased quickly'].append(round(slope.increased_quickly(i), 2))

        total_sum['increased very quickly'].append(round(slope.increased_very_quickly(i), 2))

        # # get category with highest value
        # candidate = max(total_sum, key = lambda x: sum(total_sum.get(x)))
        # truth_value = total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x))))
        # candidate_dict = total_sum
        #
        # variables.append((candidate, truth_value, candidate_dict))

        # get category with highest value
        candidate = max(total_sum, key = lambda x: sum(total_sum.get(x)))
        t_values.append(total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))))
        candidate_dict = total_sum

        variables.append((candidate, candidate_dict))

    return variables, t_values


def fuzzy_variability(series):
    '''Return linguistic variable for variability based on max value of membership function'''

    variables = []
    t_values = []

    for i in series:

        total_sum = {'very high' : [],
                     'high' : [],
                     'medium' : [],
                     'low' : [],
                     'very low' : []}

        total_sum['very high'].append(round(variability.very_high(i), 2))
        total_sum['high'].append(round(variability.high(i), 2))
        total_sum['medium'].append(round(variability.medium(i), 2))
        total_sum['low'].append(round(variability.low(i), 2))
        total_sum['very low'].append(round(variability.very_low(i), 2))

        # get category with highest value
        candidate = max(total_sum, key = lambda x: sum(total_sum.get(x)))
        t_values.append(total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x)))))
        candidate_dict = total_sum

        variables.append((candidate, candidate_dict))

    return variables, t_values

    #     # get category with highest value
    #     candidate = max(total_sum, key = lambda x: sum(total_sum.get(x)))
    #     truth_value = total_sum.get(max(total_sum, key = lambda x: sum(total_sum.get(x))))
    #     candidate_dict = total_sum
    #
    #     variables.append((candidate, truth_value, candidate_dict))
    #
    # return variables

def membership_degree(column, segments):
    '''
    1. Input df column and segments.
    2. Calculate membership degree for each attribute (trend direction, variability) for an entire summary.
    3. Return average membership degree of summary.
    '''

    # get slope of each segment and calculate corresponding fuzzy variable
    slopes = calc_slope(segments)
    _ , slope_truth_values  = fuzzy_slope(slopes)

    # get variability of each segment
    variabilities = calc_variability(column, segments)
    _ , variability_truth_values = fuzzy_variability(variabilities)

    # convert list of lists to list
    slope_truth_values = [item for sublist in slope_truth_values for item in sublist]
    variability_truth_values = [item for sublist in variability_truth_values for item in sublist]

    # calculate average membership degree
    summation = sum(slope_truth_values) + sum(variability_truth_values)
    length = len(slope_truth_values + variability_truth_values)
    avg = round(summation/length, 2)

    return avg
