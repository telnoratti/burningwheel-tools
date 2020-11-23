import pandas as pd
import numpy as np
from sympy import Function, Rational, Sum, exp, oo, series, solve, Indexed, Poly
from sympy.abc import n, x, m, i

# Old function that only works for d6 and specified expoding count
#def sym_probs(num_dice, success=4, open_ended=False):
#    initial = Rational(5/2)
#    p = initial*6**(-n)
#    g = Sum(p * x**n, (n, 0, 10)).doit()
#    print(g)
#    pol = Poly(g - 2, x)**num_dice
#    coeffs = np.array(pol.all_coeffs())
#    print(pol)
#    print(coeffs)
#    print(float(coeffs.sum()))
#    return coeffs.cumsum()[::-1]/coeffs.sum()

def pad_cut_probs(probs, length):
    probs = probs[:length]
    probs = np.pad(probs, (0, length - len(probs)), mode='constant', constant_values=(0,0)).astype(float)
    return probs

# Return a table of probabilities
def get_probs_table(die_range=(1,10,), die_faces=6, success_count=3, explode_count=0, obstacle_limit=10, log=False):
    # make die_range inclusive
    die_range = (die_range[0], die_range[1] + 1)
    columns = {'obstacle': [i for i in range(obstacle_limit + 1)]}
    col_length = obstacle_limit + 1
    for i in range(*die_range):
        column = get_probs(num_dice=i, die_faces=die_faces, success_count=success_count, explode_count=explode_count, obstacle_limit=obstacle_limit, log=log)
        column = pad_cut_probs(column, col_length)
        columns[f"{i}D"] = column
    return pd.DataFrame(data=columns).set_index('obstacle')

# Return an array of probabilties 
def get_probs(num_dice=1, die_faces=6, success_count=3, explode_count=0, obstacle_limit=10, log=False):
    # Are we using exploding dice?
    if explode_count == 0 or explode_count == None:
        return sym_probs(num_dice=num_dice, die_faces=die_faces, success_count=success_count, obstacle_limit=obstacle_limit, log=log)
    else:
        return sym_explode_probs(num_dice=num_dice, die_faces=die_faces, success_count=success_count, explode_count=explode_count, obstacle_limit=obstacle_limit, log=log)

def sym_probs(num_dice=1, die_faces=6, success_count=None, obstacle_limit=10, log=False):
    # This is so much easier when not exploding, just discreet normal
    if success_count == None:
        success_count = die_faces // 2
    # Probability of a die roll being a success
    success_chance = Rational(str(success_count) + "/" + str(die_faces))
    # Probability of a die roll failing
    failure_chance = 1 - success_chance

    if log: print(f"Die faces: {die_faces}, success count: {success_count}, success chance: {success_chance}")

    pol = Poly(failure_chance + success_chance*x, x)**num_dice
    coeffs = np.array(pol.all_coeffs())
    if log:
        print("Generating function: ", pol)
        print("Cummulative prob (low means raise obstacle limit): ", float(coeffs.sum()))
    return coeffs.cumsum()[::-1]/coeffs.sum()

def sym_explode_probs(num_dice=1, die_faces=6, success_count=None, explode_count=1, obstacle_limit=10, log=False):
    # This is so much easier when not exploding, just discreet normal
    if success_count == None:
        success_count = die_faces // 2
    # Probability of a die roll being a success
    success_chance = Rational(str(success_count) + "/" + str(die_faces))
    # Probability of a *success* exploding
    explode_chance = Rational(str(explode_count) + "/" + str(success_count))
    # Probability of a die roll failing
    failure_chance = 1 - success_chance
    # Probability of a *success* not exploding
    dud_chance = 1 - explode_chance

    if log: print(f"Die faces: {die_faces}, success count: {success_count}, success chance: {success_chance}, explode count: {explode_count}, explode_chance: {explode_chance}")

    # There is probably a way to do this that doesn't involve calculating the
    # first two terms and finding the ratio this is how I did it by hand though
    # and I don't remember enough combinatorics to know a better way
    p_0 = failure_chance
    p_1 = success_chance*(dud_chance + explode_chance*failure_chance)
    p_2 = success_chance*explode_chance*success_chance*(dud_chance + explode_chance*failure_chance)
    if log:
        print("p_0 = ", p_0)
        print("p_1 = ", p_1)
        print("p_2 = ", p_2)

    term_relation = p_2 / p_1

    # We will need p_0 to be the failure chance, that doesn't always work out, so we force it
    # I don't really know why this works, but it does in the BW case
    initial = p_1 / term_relation

    correction = p_0 - initial
    if log:
        print("term relation = ", term_relation)
        print("initial = ", initial)
        print("correction = ", correction)

    p_n = initial*(success_chance * explode_chance)**(n)
    g = Sum(p_n * x**n, (n, 0, obstacle_limit)).doit()
    pol = Poly(g + correction, x)**num_dice
    coeffs = np.array(pol.all_coeffs())
    if log:
        print("Generating function: ", pol)
        print("Cummulative prob (low means raise obstacle limit): ", float(coeffs.sum()))
    return coeffs.cumsum()[::-1]/coeffs.sum()

#print(list(map(float,sym_probs(2))))
#print(list(map(float,get_probs(num_dice=3))))
#print(list(map(float,get_probs(num_dice=3, explode_count=1))))
#print(list(map(float,sym_explode_probs(num_dice=3))))
#print(list(map(float,sym_probs_flex(1, die_faces=4))))
#print(list(map(float,sym_probs_flex(1, success_count=4))))

# Given N dice to roll, return number of successes
def explode(N, explode_min=6, success=4):
    if N == 0:
        return 0
    rolls = np.random.randint(1,7, size=N)
    exploders = (rolls >= explode_min).sum()
    successes = (rolls >= success).sum()
    return successes + explode(exploders, explode_min=explode_min, success=success)

# This function produces the probabilities using simulation, only works for burning wheel dice system right now
# I do not recommend using this as it took 10m iterations to get an accuracy of +/- 0.001, but it's good to verify the symbolic version
def simulate_probs(num_dice, success=4, open_ended=False, iterations=10000):
    rolls = np.random.randint(1, 7, size=(iterations, num_dice))
    print(rolls)
    successes = (rolls >= success).sum(axis=1)
    # Reroll 6
    if open_ended:
        sixes = (rolls >= 6).sum(axis=1)
        additional_successes = np.vectorize(explode)(sixes)
        successes = successes + additional_successes

    freq_count = np.bincount(successes)
    print(freq_count/freq_count.sum())
    return freq_count[::-1].cumsum()[::-1]/freq_count.sum()

#np.random.seed(0)
#print(get_probs(1))
#print(get_probs(1, open_ended=True, success=4, iterations=100000))
#
#np.random.seed(0)
#print(explode(12))
#print(get_probs_table(explode_count=1))
#print(get_probs_table(explode_count=0))
