import sympy
from sympy import rsolve, Function, expand, Rational, Sum, Poly
from sympy.abc import *

from sympy import O
import numpy as np
from numpy.polynomial import polynomial as np_poly

### Recurrence relations precomputed, needed for exploding cases
# These are kept as recurrence relations because I get an error when trying to
# find the recurrence relation before substituting in values. If I were a math
# wiz I'd just solve it myself, but I get tripped up tracking all the variables
## exploding
#D_e = Function('D_e')
#R_e = D_e(n+1) - s*e*D_e(n)
D_e_1 = s*e*(d/e+f)
D_e_0 = f
# Solve for D_e_n after substitution with 
#   D_e_n = rsolve(R_e, D_e(n), {D_e(1): D_e_1})
#   P_r_e_n = D_e_0 + Sum(D_e_n*x**n, (n, 1, N)) + O(x**(N+1))
# Calculated with Mathematica which has a more robust recurrence solver
P_e = (d+e*f)*(e*s)**n/e

# These initial conditions ignore all rolls with at least one failure used for rerolling one die
D_e_s_1 = s*d
D_e_s_0 = 0
# Calculated with Mathematica which has a more robust recurrence solver
P_e_s = d*(e*s)**n/e

## reroll failures, exploding
#D_r_e   = Function('D_r_e')
# The second term is derived from the exploding non-reroll case
#R_r_e   = D_r_e(n+1) - f*(s*e)**(n+1)*(d/e+f) - s*e*D_r_e(n)
D_r_e_1 = f*s*(d+e*f) + s*(d+e*f*f)
D_r_e_0 = f**2
# Calculated with Mathematica which has a more robust recurrence solver
P_r_e = (e*s)**n*(e*f**2*n + d*f*n + e*f**2 + d)/e
# Solve for P_r_e after substitution with 
#   D_r_e_n = rsolve(R_r_e, D_r_e(n), {D_r_e(1): D_r_e_1})
#   P_r_e = D_r_e_0 + Sum(D_r_e_n*x**n, (n, 1, N)) + O(x**(N+1))
# The case of only successes
#D_r_e_s   = Function('D_r_e_s')
#R_r_e_s   = D_r_e_s(n+1) - f*d*(e*s)**n/e - s*e*D_r_e_s(n)
D_r_e_s_1 = f*s*d+s*d
D_r_e_s_0 = 0
# Calculated with Mathematica which has a more robust recurrence solver
# RSolve[{p[n+1] == f*d*(s*e)^(n+1)/e + s*e*p[n], p[1] == f*s*d + s*d}, p, n]
P_r_e_s = (e*s)**n * d*(1+f*n)/e

### Polynomials precomputed, for use in non-exploding cases
## Simple die roll
P_x = f + s*x
P_0 = f
P_1 = s
# P is just P_x after substitution
# only success
P_s_0 = 0
P_s_1 = s

## Die roll and reroll one failure
P_r_x = f**2 + (1 - f**2)*x
P_r_0 = f**2
P_r_1 = (1-f**2)

P_r_s_0 = 0
P_r_s_1 = f*s+s

# This is a convienence function for rolling some dice for burning wheel
def roll_dice(num_dice=1, shade='black', open_ended=False, luck=False, boon=0, divine_inspiration=False, saving_grace=False, high_ob=10, cum_sum=False, log=False):
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

    # Numpy method is most effecient
    p_1 = die_poly(die_faces=6, success_count=success_count, explode_count=explode_count, reroll=saving_grace, order=high_ob)
    p_n = p_1
    for i in range(1,num_dice):
        p_n = np_poly.polymul(p_n, p_1)[:high_ob+1]

    # Now p_n is an array of polynomial coeffecients from low to high which is the generating function for our die
    if reroll_one:
        # I needs the odds of at least one failure, 
        # For two dice this should be 1, 1, 8/9, ...

        # s_n's mth term is the odds of rolling m successes without any dice failing
        s_1 = die_poly(die_faces=6, success_count=success_count, explode_count=explode_count, reroll=saving_grace, only_success=True, order=high_ob)
        s_n = s_1
        for i in range(1,num_dice):
            s_n = np_poly.polymul(s_n, s_1)[:high_ob+1]

        s_n = s_n
        # f_n is the chance of getting 
        f_n = (1 - s_n)
        # f_n is the odds of getting to roll an additional die from luck as is a mask for our luck die
        #print(list(map(float,p_n)))
        #print(sum(list(map(float,p_n))))
        #print(list(map(float,p_n)))
        #print(sum(list(map(float,p_n))))
        fl = lambda x: list(map(float,x))
        p_nr_1 = die_poly(die_faces=6, success_count=success_count, explode_count=explode_count, reroll=False, order=high_ob)
        p_n_r = np_poly.polymul((p_n - s_n), p_nr_1)[:high_ob+1]
        p_n = s_n + p_n_r
        #print(list(map(float,p_n)))
        #print(sum(list(map(float,p_n))))
        if log:
            print("Odds of n successes including a failure: ", fl(f_n))
            print("Odds of n successes having no failures : ", fl(s_n))
            print("Odds of only success (1d): ", fl(s_1))

    results = np.zeros(high_ob+1, dtype=object)
    results[:p_n.shape[0]] = p_n

    if cum_sum:
        results = results[::-1].cumsum()[::-1] + (1 - results.sum())

    return results

def pad_cut_probs(probs, length):
    probs = probs[:length]
    probs = np.pad(probs, (0, length - len(probs)), mode='constant', constant_values=(0,0)).astype(int)
    return probs

# Yields a numpy array length order+1 of the coeffecients of the generating function from low to high
# only_success calculates the probability of a value with no terminating failures
def die_poly(die_faces=6, success_count=3, explode_count=0, reroll=False, only_success=False, order=10):
    # Probability of a die roll being a success
    success_chance = Rational(str(success_count) + "/" + str(die_faces))
    # Probability of a *success* exploding
    explode_chance = Rational(str(explode_count) + "/" + str(success_count))

    # Have our substitutions as a short hand
    def sp(x):
        if isinstance(x, sympy.core.expr.Expr):
            return x.subs(f, 1 - s).subs(s, success_chance).subs(d, 1 - e).subs(e, explode_chance)
        return x

    coeffs = np.zeros(order+1, dtype=object)
    if explode_count > 0:
        # We have an exploding die
        if reroll:
            # We are rerolling each failure once
            F = None
            if only_success:
                #F = rsolve(sp(R_r_e_s), D_r_e_s(n), {D_r_e_s(1): sp(D_r_e_s_1)})
                F = sp(P_r_e_s)
                coeffs[0] = sp(D_r_e_s_0)
            else:
                #F = rsolve(sp(R_r_e), D_r_e(n), {D_r_e(1): sp(D_r_e_1)})
                F = sp(P_r_e)
                coeffs[0] = sp(D_r_e_0)

            for i in range(1, order + 1):
                coeffs[i] = F.subs(n,i)
        else:
            # We have the simple exploding die formula
            F = None
            if only_success:
                F = sp(P_e_s)
                coeffs[0] = sp(D_e_s_0)
            else:
                F = sp(P_e)
                coeffs[0] = sp(D_e_0)
            for i in range(1, order + 1):
                coeffs[i] = F.subs(n,i)
    else:
        # We have a non-exploding die
        if reroll:
            # We must reroll the first failure
            if only_success:
                coeffs[0] = sp(P_r_s_0)
                coeffs[1] = sp(P_r_s_1)
            else:
                coeffs[0] = sp(P_r_0)
                coeffs[1] = sp(P_r_1)

        else:
            if only_success:
                coeffs[0] = sp(P_s_0)
                coeffs[1] = sp(P_s_1)
            else:
                coeffs[0] = sp(P_0)
                coeffs[1] = sp(P_1)
    return coeffs
