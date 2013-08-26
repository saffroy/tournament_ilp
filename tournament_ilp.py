#!/usr/bin/env python

import sys, getopt

from numpy import *
from openopt import MILP
from glpk.glpkpi import *

'''
Goal:
  Compute a schedule for a local sports league with N teams (eg. 10)

T teams
D days
P places
N games per team per day at most

Rules:
Team i plays team j exactly twice, on different days (and in different places?)
Team i plays all games on day d in the same place

---
Team i plays "home" game against team j on day d at place p <=> H(i,j,d,p) = 1
Team i is at place p on day d <=> P(i,d,p) = 1

Team i never plays itself:
 do not define H(i,i,*,*)

Team i plays home game and return game exactly once with each team:
 For all i, j: Sum(d, Sum(p, H(i,j,d,p))) = 1

Team i is exacly one place on a given day d:
 For all i, d: Sum(p, P(i,d,p)) = 1

Team i plays each of its games on day d at a place it is:
 For all i, j, d, p: H(i,j,d,p) <= P(i,d,p) && H(i,j,d,p) <= P(j,d,p) 

Team i does not play home and return game with team j on same day:
 For all i, j, d: Sum(p, H(i,j,d,p) + H(j,i,d,p)) <= 1

Team i plays at most N games per day:
 For all i, d: Sum(j, Sum(p, H(i,j,d,p) + H(j,i,d,p))) <= N

Site p hosts at most M games per day:
 For all p, d: Sum(i, Sum(j, H(i,j,d,p) + H(j,i,d,p))) <= M

---
Inspiration
  http://www.openopt.org/MILP
  http://trac.openopt.org/openopt/browser/PythonPackages/OpenOpt/openopt/examples/milp_1.py
  http://www.pitt.edu/~dsaure/papers/OR_Chilean_soccer.pdf

'''

N_teams = 10
N_places = N_teams
max_games_per_team = 2
max_games_per_place = 5
N_days = ((N_teams - 1) * 2) / max_games_per_team + 2

def update_config():
    global H_vars, P_vars, N_vars

    H_vars = N_teams * (N_teams - 1) * N_days * N_places
    P_vars = N_teams * N_days * N_places
    N_vars = H_vars + P_vars

def H(i,j,d,p):
    if i > j:
        # diagonal H(i,i,*,*) doesn't exist
        i -= 1
    return i + (N_teams - 1) * (j + N_teams * (d + N_days * p))

def P(i,d,p):
    return H_vars + i + N_teams * (d + N_days * p)

def teamPairs():
    # enumerate valid pairs of opponent teams
    for i in xrange(N_teams):
        for j in xrange(N_teams):
            if i != j:
                yield (i, j)

def make_model():
    '''Build a model of the problem in the form of a set of linear constraints'''

    global A_eq, b_eq, A_lt, b_lt, f
    A_eq = []
    b_eq = []
    A_lt = []
    b_lt = []

    ''' Team i plays home game and return game exactly once with each team:
        For all i, j: Sum(d, Sum(p, H(i,j,d,p))) = 1 '''
    for i, j in teamPairs():
        r = {}
        for d in xrange(N_days):
            for p in xrange(N_places):
                r[H(i,j,d,p)] = 1
        A_eq.append(r)
        b_eq.append(1)

    '''Team i is exacly one place on a given day d:
       For all i, d: Sum(p, P(i,d,p)) = 1 '''
    for i in xrange(N_teams):
        for d in xrange(N_days):
            r = {}
            for p in xrange(N_places):
                r[P(i,d,p)] = 1
            A_eq.append(r)
            b_eq.append(1)

    '''Team i does not play home and return game with team j on same day:
       For all i, j, d: Sum(p, H(i,j,d,p) + H(j,i,d,p)) <= 1 '''
    for i, j in teamPairs():
        if i > j:
            continue
        for d in xrange(N_days):
            r = {}
            for p in xrange(N_places):
                r[H(i,j,d,p)] = 1
                r[H(j,i,d,p)] = 1
            A_lt.append(r)
            b_lt.append(1)

    '''Team i plays each of its games on day d at a place it is:
       For all i, j, d, p: H(i,j,d,p) <= P(i,d,p) && H(i,j,d,p) <= P(j,d,p) '''
    for i, j in teamPairs():
        for d in xrange(N_days):
            for p in xrange(N_places):
                r = {}
                r[H(i,j,d,p)] = 1
                r[P(i,d,p)] = -1
                A_lt.append(r)
                b_lt.append(0)

                r = {}
                r[H(i,j,d,p)] = 1
                r[P(j,d,p)] = -1
                A_lt.append(r)
                b_lt.append(0)

    '''Team i plays at most N games per day:
       For all i, d: Sum(j, Sum(p, H(i,j,d,p) + H(j,i,d,p))) <= N '''
    for i in xrange(N_teams):
        for d in xrange(N_days):
            r = {}
            for j in xrange(N_teams):
                if i != j:
                    for p in xrange(N_places):
                        r[H(i,j,d,p)] = 1
                        r[H(j,i,d,p)] = 1
            A_lt.append(r)
            b_lt.append(max_games_per_team)

    '''Site p hosts at most M games per day:
       For all p, d: Sum(i, Sum(j, H(i,j,d,p) + H(j,i,d,p))) <= M'''
    for d in xrange(N_days):
        for p in xrange(N_places):
            r = {}
            for i, j in teamPairs():
                r[H(i,j,d,p)] = 1
                r[H(j,i,d,p)] = 1
            A_lt.append(r)
            b_lt.append(max_games_per_place)

    '''Minimize number of games played on last days'''
    f = {}
    for i, j in teamPairs():
        for p in xrange(N_places):
            for d in xrange(N_days):
                f[H(i,j,d,p)] = d

    global N_eq, N_lt, N_cons
    N_eq = len(A_eq)
    N_lt = len(A_lt)
    N_cons = N_eq + N_lt

def print_sol(res):
    print 'Game schedule:'
    for i, j in teamPairs():
        for d in xrange(N_days):
            for p in xrange(N_places):
                if res[H(i,j,d,p)] > 0:
                    print 'Team %2d vs. %2d day %2d place %2d' % (i,j,d,p)

def compute_sol():
    intVars = range(N_vars)
    lb = zeros(N_vars)
    ub = ones(N_vars)

    def denseRow(r):
        row = zeros(N_vars)
        for i, v in r.items():
            row[i] = v
        return row

    A = map(denseRow, A_lt)
    Aeq = map(denseRow, A_eq)
    ff = denseRow(f)

    p = MILP(f=ff, lb=lb, ub=ub, A=A, b=b_lt, Aeq=Aeq, beq=b_eq, intVars=intVars, goal='min')
    r = p.solve('glpk', iprint =-1)
    return r.xf

def save_model(path, mps=True):
    prob = glp_create_prob()
    glp_set_prob_name(prob, "tournament scheduling")
    glp_set_obj_dir(prob, GLP_MIN)

    glp_add_cols(prob, N_vars)
    for i in xrange(N_vars):
        glp_set_col_kind(prob, i+1, GLP_BV)
    for i, v in f.items():
        glp_set_obj_coef(prob, i+1, v)

    glp_add_rows(prob, N_cons)
    ind = intArray(N_vars + 1)
    val = doubleArray(N_vars + 1)

    def add_row(j, r, type, lb, ub):
        n_cols = 0
        for i, v in r.items():
            n_cols += 1
            ind[n_cols] = i+1
            val[n_cols] = v
        glp_set_mat_row(prob, j+1, n_cols, ind, val)
        glp_set_row_bnds(prob, j+1, type, lb, ub)

    for j, r, b in zip(xrange(N_eq), A_eq, b_eq):
        add_row(j, r, GLP_FX, b, b)
    for j, r, b in zip(xrange(N_lt), A_lt, b_lt):
        add_row(N_eq + j, r, GLP_UP, 0, b)

    if mps:
        glp_write_mps(prob, GLP_MPS_FILE, None, path)
    else:
        glp_write_lp(prob, None, path)

def parse_sol(path):
    res = zeros(N_vars)
    with open(path) as f:
        for line in f:
            if '*' in line:
                words = line.split()
                res[int(words[0])-1] = int(words[3])
    return res

def usage():
    print 'usage: %s [options]' % sys.argv[0]
    print '-h 		print this help'
    print '-t <teams>	number of teams (default: %d)' % N_teams
    print '-d <days>	number of days (default: %d)' % N_days
    print '-p <places>	number of places (default: %d)' % N_places
    print '-g <games>	max number of games per day per team (default: %d)' % max_games_per_team
    print '-G <games>	max number of games per day per place (default: %d)' % max_games_per_place
    print '-s <model>	save problem to MPS model file'
    print '-D <solution>	display solution from glpsol output file'
    print 'Default action is to try to find a solution, except with options -s and -D.'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:d:p:g:G:s:D:", [])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(1)

    global N_teams, N_days, N_places, max_games_per_team, max_games_per_place
    model_path = None
    solution_path = None

    for o, a in opts:
        if o == '-h':
            usage()
            sys.exit(0)
        elif o == '-t':
            N_teams = int(a)
        elif o == '-d':
            N_days = int(a)
        elif o == '-p':
            N_places = int(a)
        elif o == '-g':
            max_games_per_team = int(a)
        elif o == '-G':
            max_games_per_place = int(a)
        elif o == '-s':
            model_path = a
        elif o == '-D':
            solution_path = a
        else:
            assert False, "unhandled option"

    if model_path and solution_path:
        print 'error: options -d and -s are mutually exclusive'
        sys.exit(1)

    update_config()
    make_model()

    print 'Team: %d Days: %d' % (N_teams, N_days)
    print 'Vars: %d Constraints: %d (eq: %d lt: %d) Non-zero ceofs: %d' % (
        N_vars, N_cons, N_eq, N_lt, sum(map(len, A_eq + A_lt)))

    if model_path:
        print 'Saving problem to model file %s' % model_path
        save_model(model_path, mps=True)
    elif solution_path:
        print 'Reading solution from file %s' % solution_path
        sol = parse_sol(solution_path)
        print_sol(sol)
    else:
        sol = compute_sol()
        print_sol(sol)

if __name__ == "__main__":
    main()
