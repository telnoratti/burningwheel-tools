import itertools
from collections import Counter
import pandas as pd
import numpy as np
from multiprocessing import Pool, Process, Queue

from gen_function import roll_dice

NTHREADS=6

def pad_cut_probs(probs, length):
    probs = probs[:length]
    probs = np.pad(probs, (0, length - len(probs)), mode='constant', constant_values=(0,0)).astype(float)
    return probs

# A flexible function to return a function that will effeciently generate random numbers
def roller(faces=6):
    SIZE = 100000
    rolls = np.random.randint(1, faces + 1, size=SIZE)
    i = -1
    def get_a_roll():
        nonlocal i, rolls
        if i >= SIZE - 1:
            rolls = np.random.randint(1, faces + 1, size=SIZE)
            i = -1
        i += 1
        return rolls[i]

    return get_a_roll

# To test our math, let's compare against a simulation
# Function returns number of successes from a single die roll
# reroll_fail is the number of failures to reroll
def roll_die(roll_func, rolls, success=4, explode=6, reroll_fail=0):
    roll = roll_func()

    if roll < success:
        # Roll failed
        if reroll_fail == 0:
            rolls.append(roll)
            return 0
        else:
            return roll_die(roll_func, rolls, success=success, explode=explode, reroll_fail=reroll_fail - 1)
    else:
        # Success
        if roll >= explode:
            # Explode the die
            rolls.append(roll)
            return 1 + roll_die(roll_func, rolls, success=success, explode=explode, reroll_fail=reroll_fail)
        else:
            rolls.append(roll)
            return 1

# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sim_roll_dice(num_dice=1, shade='black', open_ended=False, luck=False, boon=0, divine_inspiration=False, saving_grace=False, high_ob=10, trials=100000, log=False):
    # What counts as a success
    shade_to_faces = {
            'black': 3,
            'grey': 4,
            'white': 5,
    }
    success_count = shade_to_faces[shade]

    # Check the open-ended/luck behaviour
    explode_count = 0
    reroll_one = False
    if open_ended:
        explode_count = 1
        if luck:
            reroll_one = True
    elif luck:
        explode_count = 1

    # Boon and divine_inspiration impact number of dice rolled
    if divine_inspiration:
        num_dice = num_dice * 2

    num_dice += boon

    reroll_fail = 0
    if saving_grace:
        reroll_fail = 1

    # Do the simulation
    roll_func = roller()
    totals = []
    pre_results = []
    included_failure = []
    for i in range(trials):
        rolls = []
        total = 0
        for i in range(num_dice):
            total += roll_die(roll_func, rolls, success=(6+1 - success_count), explode=(6+1-explode_count), reroll_fail=reroll_fail)
        # Reroll a die for luck
        pre_results.append(total)
        if reroll_one:
            # Check if any fail, then reroll one die for that one.
            if min(rolls) < 6+1-success_count:
                included_failure.append(total)
                total += roll_die(roll_func, rolls, success=(6+1 - success_count), explode=(6+1-explode_count), reroll_fail=0)
        totals.append(total)

    results = np.array(totals)
    odds = np.bincount(results)/trials

    #if reroll_one:
    #    pre_results_r = pad_cut_probs(np.bincount(np.array(pre_results)), 11)
    #    failure_r = pad_cut_probs(np.bincount(np.array(included_failure)), 11)
    #    post_results_r = pad_cut_probs(np.bincount(np.array(totals)), 11)
    #    odds_r = pad_cut_probs(odds, 11)
    #    fl = lambda x: list(map(float,x))
    #        with np.errstate(divide='ignore', invalid='ignore'):
    #            print("pre_results_r: ", fl(pre_results_r))
    #            print("failure_r: ", fl(failure_r))
    #            print("post_results_r: ", fl(post_results_r))
    #            print("odds_r: ", fl(odds_r))
    #            print("Reroll odds: ", fl(post_results_r/pre_results_r))
    #            print("Reroll to non-reroll diff: ", fl((post_results_r - pre_results_r)/pre_results_r))
    #            print("Odds of n successes including a failure: ",fl(failure_r/pre_results_r))

    return odds

params = {
        'num_dice': list(range(1,4)),
        'shade': ['black', 'grey', 'white'],
        'open_ended': [True, False],
        'luck': [True, False],
        'divine_inspiration': [True, False],
        'saving_grace': [True, False],
        }
#params = {
#        'num_dice': [1],
#        'shade': ['black'], #, 'grey', 'white'],
#        'open_ended': [True],
#        'luck': [True],
#        'divine_inspiration': [False],
#        'saving_grace': [True],
#        }

procs = []

p_params = product_dict(**params)
def sim_roll_dice_d(kwargs):
    sim = sim_roll_dice(**kwargs)
    sim = pad_cut_probs(sim, 11)
    exact = roll_dice(**kwargs)
    exact = pad_cut_probs(exact, 11)
    #print(float(sum(exact)))
    diff = sim - exact
    if max(diff) > 1e-02:
        print(kwargs)
        print("Simulated: ", list(map(float,sim)))
        print("Exact: ", list(map(float,exact)))
        print("Diff: ", list(map(float,sim-exact)))

with Pool(processes=NTHREADS) as pool:
    pool.map(sim_roll_dice_d, p_params)

for p in procs:
    p.join()

import sys
sys.exit(1)
