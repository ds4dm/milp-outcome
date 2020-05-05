import numpy as np
import pandas as pd
import argparse
import math
import sys
import os
import glob
from collections import OrderedDict

import shared

"""
Process data collected with callbacks during (partial) B&B run (data_run) into a single array of features.
Build a data point of (features, label).

Each stored npz file contains 

    "ft_matrix"     : matrix of raw data from BranchCB and NodeCB
    "name"          : problem name
    "seed"          : the seed of the computation
    "dataset"       : name of the dataset to which the problem belongs
    "rho"           : rho
    "tau"           : tau
    "node_limit"    : node limit used in data collection
    "label_time"    : time with respect to which a label was computed
    "label"         : the computed label in {0, 1}
    "trivial"       : boolean flag
    "sol_time"      : time of resolution (or timelimit)
    "data_final_info: array of final information on the run (useful to check trivial instances)
    
"""


"""
Parser and auxiliary functions
"""

parser = argparse.ArgumentParser(description='Arg parser for features script.')

# Paths
parser.add_argument(
    '--npy_path',
    type=str,
    required=True,
    help='Path to directory of npz files to be processed (usually */NPY).'
)
parser.add_argument(
    '--learning_path',
    type=str,
    required=True,
    help='Path to directory where data ready for learning will be saved.'
)
parser.add_argument(
    '--inst_path',
    type=str,
    default=shared.INST_PATH,
    help='Path to directory containing original instances. Can be set in shared.py.'
)

# Filename of generated data
parser.add_argument(
    '--filename',
    type=str,
    required=True,
    help='Name for filename.npz to be saved with dataset of (features, label) pairs.'
)

ARGS = parser.parse_args()


""" Routines """


# NOTE: this is *not* the same definition as the MIP Relative Gap implemented by CPLEX
def primal_dual_gap(best_ub, best_lb):
    """
    :param best_ub: current best upper bound: c^Tx~
    :param best_lb: current best lower bound: z_
    """
    if (abs(best_ub) == 0) & (abs(best_lb) == 0):
        return 0
    elif best_ub * best_lb < 0:
        return 1
    else:
        return abs(best_ub - best_lb) / max([abs(best_ub), abs(best_lb)])


def primal_gap(best_sol, feas_sol):
    """
    :param best_sol: optimal or best known solution value: c^Tx*,
    :param feas_sol: feasible solution value: c^Tx~
    """
    if (abs(best_sol) == 0) & (abs(feas_sol) == 0):
        return 0
    elif best_sol * feas_sol < 0:
        return 1
    else:
        return abs(best_sol - feas_sol) / max([abs(best_sol), abs(feas_sol)])


def get_chunks(node_df, branch_df):
    """
    Subdivision of BranchCB data (branch_df) in up to 4 chunks (using 25-50-75% of eta nodes)
    Returns:
        idx_split: global_cb_count splits for the chunks
        chunk_num: number of *complete* chunks
        num_nodes_list: list containing the number of processed nodes in each chunk
    NOTE: last chunk might appear not completed in runs that solved before tau!
    In general, this computations should be considered as approximations of the different phases.
    """
    idx_split = list(node_df['global_cb_count'].values)
    idx_split.insert(0, branch_df.iloc[0]['global_cb_count'])  # add starting idx (1)

    if len(idx_split) < 5:
        chunk_num = len(idx_split) - 1  # less than 4 complete chunks, add last collected idx from branch_df (end)
        idx_split.append(branch_df.iloc[-1]['global_cb_count'])
    else:
        chunk_num = 4  # no need to add idx for the end of the run

    chunk_nodes_list = list()  # contains total # nodes processed at the end of each chunk
    for i in range(len(idx_split) - 1):
        tmp_df = branch_df.loc[(branch_df['global_cb_count'] > idx_split[i]) &
                               (branch_df['global_cb_count'] <= idx_split[i + 1])]
        # check if tmp_df is empty
        if tmp_df.shape[0] == 0:
            chunk_nodes_list.append(0)
        else:
            chunk_nodes_list.append(tmp_df.iloc[-1]['num_nodes'])

    num_nodes_list = list()  # contains the deltas of nodes processed within each chunk
    num_nodes_list.append(chunk_nodes_list[0])
    for i in range(len(chunk_nodes_list) - 1):
        num_nodes_list.append(chunk_nodes_list[i + 1] - chunk_nodes_list[i])

    return idx_split, chunk_num, num_nodes_list


"""
Features from last rows of DataFrames
"""


def from_last_seen_data(last_branch, last_node):
    """
    nodeID / num_nodes
    nodeID / nodes_left
    nodes_left, best_integer, best_bound, itCnt, gap, has_incumbent, num_nodes
    objective / best_integer, best_bound / objective, best_bound / best_integer
    open_nodes_len, open_nodes_max, open_nodes_min, open_nodes_avg
    num_nodes_at_max / open_nodes_len, num_nodes_at_min / open_nodes_len
    open_nodes_max / best_integer, open_nodes_min / best_integer, open_nodes_min / open_nodes_max
    objective / open_nodes_max, open_nodes_min / objective
    num_cuts
    """

    last_data_list = []

    first = last_branch['node'] / float(last_branch['num_nodes']) if last_branch['num_nodes'] != 0 \
        else None
    second = last_branch['node'] / float(last_branch['nodes_left']) if last_branch['nodes_left'] != 0 \
        else None

    last_data_list.append(first)
    last_data_list.append(second)

    last_data_list.append(last_branch['nodes_left'])
    last_data_list.append(last_branch['best_integer'])
    last_data_list.append(last_branch['best_bound'])
    last_data_list.append(last_branch['itCnt'])
    last_data_list.append(last_branch['gap'])
    last_data_list.append(last_branch['has_incumbent'])
    last_data_list.append(last_branch['num_nodes'])

    third = last_branch['objective'] / last_branch['best_integer'] \
        if last_branch['best_integer'] and last_branch['objective'] else None
    fourth = last_branch['best_bound'] / last_branch['objective'] \
        if last_branch['best_bound'] and last_branch['objective'] else None
    fifth = last_branch['best_bound'] / last_branch['best_integer'] \
        if last_branch['best_integer'] and last_branch['best_bound'] else None

    last_data_list.append(third)
    last_data_list.append(fourth)
    last_data_list.append(fifth)

    last_data_list.append(last_node['open_nodes_len'])
    last_data_list.append(last_node['open_nodes_max'])
    last_data_list.append(last_node['open_nodes_min'])
    last_data_list.append(last_node['open_nodes_avg'])
    last_data_list.append(last_node['num_nodes_at_max'] / float(last_node['open_nodes_len']))
    last_data_list.append(last_node['num_nodes_at_min'] / float(last_node['open_nodes_len']))

    # open_nodes_max/best_integer, open_nodes_min/best_integer, open_nodes_min/open_nodes_max

    sixth = last_node['open_nodes_max'] / float(last_branch['best_integer']) \
        if last_branch['best_integer'] and last_node['open_nodes_max'] else None
    seventh = last_node['open_nodes_min'] / float(last_branch['best_integer']) \
        if last_branch['best_integer'] and last_node['open_nodes_min'] else None
    eighth = last_node['open_nodes_min'] / float(last_node['open_nodes_max']) \
        if last_node['open_nodes_max'] and last_node['open_nodes_min'] else None

    last_data_list.append(sixth)
    last_data_list.append(eighth)
    last_data_list.append(seventh)

    # objective/open_nodes_max, open_nodes_min/objective

    ninth = last_branch['objective'] / float(last_node['open_nodes_max']) \
        if last_branch['objective'] and last_node['open_nodes_max'] else None
    tenth = last_node['open_nodes_min'] / float(last_branch['objective']) \
        if last_branch['objective'] and last_node['open_nodes_min'] else None

    last_data_list.append(ninth)
    last_data_list.append(tenth)

    last_data_list.append(last_node['num_cuts'])

    if len(last_data_list) != 24:
        print("*** len(last_seen_data): {}".format(len(last_data_list)))

    return last_data_list, len(last_data_list)


"""
Other features
"""


def about_pruned(branch_df):
    """
    total number of pruned nodes
    pruned throughput: pruned / num_nodes
    pruned / nodes_left
    """
    pruned = pd.Series(branch_df['num_nodes'].diff(1)-1)  # correction!
    cumulative = float(pruned.sum())
    second = cumulative / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    third = cumulative / branch_df['nodes_left'].iloc[-1] if branch_df['nodes_left'].iloc[-1] != 0 else None

    pruned_list = [cumulative, second, third]

    if len(pruned_list) != 3:
        print("*** len(pruned_list): {}".format(len(pruned_list)))

    return pruned_list, len(pruned_list)


def about_nodes_left(branch_df, idx_split, num_nodes_list):
    """
    nodes_left throughput (w.r.t. last seen node)
    nodes_left at last seen node / max nodes_left
    4x rate of change (slopes) of nodes_left in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
    """
    nodes_left_list = []
    slopes = [None]*4

    first = branch_df['nodes_left'].iloc[-1] / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 \
        else None

    nodes_left_list.append(first)
    nodes_left_list.append(branch_df.iloc[-1]['nodes_left'] / branch_df['nodes_left'].max())

    for i in range(len(idx_split) - 1):
        tmp_df = branch_df.loc[(branch_df['global_cb_count'] > idx_split[i]) &
                               (branch_df['global_cb_count'] < idx_split[i + 1])]

        # check if tmp_df is empty
        if tmp_df.shape[0] == 0:
            slopes[i] = 0
        else:
            slopes[i] = (tmp_df['nodes_left'].iloc[-1] - tmp_df['nodes_left'].iloc[0]) / float(num_nodes_list[i]) \
                if num_nodes_list[i] else 0
    nodes_left_list.extend(slopes)

    if len(nodes_left_list) != 6:
        print("*** len(nodes_left_list): {}".format(len(nodes_left_list)))

    return nodes_left_list, len(nodes_left_list)


def about_iinf(branch_df, num_discrete_vars):
    """
    max, min, avg iinf / total number of discrete variables (integer + binary)
    last seen iinf / total number of discrete variables (integer + binary)
    # nodes with iinf in 0.05 quantile / total number of calls
    last iinf / value of 0.05 quantile
    distance between last seen node in 0.05 quantile to end of data collection / total number of calls (rows)
    """
    # num_discrete_vars must be != 0
    quantile_df = branch_df.loc[branch_df['iinf'] < branch_df['iinf'].quantile(.05)][['iinf', 'times_called']]

    iinf_list = [
        branch_df['iinf'].max() / float(num_discrete_vars),
        branch_df['iinf'].min() / float(num_discrete_vars),
        branch_df['iinf'].mean() / float(num_discrete_vars),
        branch_df['iinf'].iloc[-1] / float(num_discrete_vars),
        quantile_df.shape[0] / float(branch_df.times_called.max())
    ]

    first = branch_df['iinf'].iloc[-1] / float(branch_df['iinf'].quantile(.05)) \
        if branch_df['iinf'].quantile(.05) else None

    iinf_list.append(first)
    iinf_list.append((branch_df.times_called.max() - quantile_df.times_called.max())
                     / float(branch_df.times_called.max()))

    if len(iinf_list) != 7:
        print("*** len(iinf_list): {}".format(len(iinf_list)))

    return iinf_list, len(iinf_list)


def about_itcnt(branch_df):
    """
    itcnt throughput
    """
    first = branch_df['itCnt'].iloc[-1] / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 \
        else None
    itcnt_list = [first]

    if len(itcnt_list) != 1:
        print("***len(itcnt_list): {}".format(len(itcnt_list)))

    return itcnt_list, len(itcnt_list)


def about_integer_feasible(branch_df):
    """
    # of integer feasible found / total # of processed nodes
    incumbent found before integer feasible node (boolean)
    depth first incumbent
    """

    if branch_df.loc[branch_df['has_incumbent'] == 1].shape[0] == 0:
        int_feas_list = [
            branch_df.loc[branch_df['is_integer_feasible'] == 1].shape[0] / branch_df.iloc[-1]['num_nodes'],
            False,
            None
        ]
    else:
        first_inc_row = branch_df.loc[branch_df['has_incumbent'] == 1][
            ['global_cb_count', 'times_called', 'is_integer_feasible', 'has_incumbent', 'depth']].iloc[0]

        first = branch_df.loc[branch_df['is_integer_feasible'] == 1].shape[0] / branch_df.iloc[-1]['num_nodes'] \
            if branch_df.iloc[-1]['num_nodes'] else None

        second = True if \
            (not first_inc_row['is_integer_feasible']) & (first_inc_row['has_incumbent']) \
            else False

        int_feas_list = [
            first,
            second,
            first_inc_row['depth']
        ]

    if len(int_feas_list) != 3:
        print("***len(int_feas_list): {}".format(len(int_feas_list)))

    return int_feas_list, len(int_feas_list)


def about_incumbent(branch_df):
    """
    number of incumbent updates
    incumbent throughput: num_updates / num_nodes
    max_improvement, min_improvement, avg_improvement
    avg incumbent improvement / first incumbent value
    max, min, avg distance between past incumbent updates
    distance between last update and last node explored
    """

    abs_improvement = pd.Series(abs(branch_df['best_integer'].diff(1)))
    bool_updates = pd.Series((abs_improvement != 0))
    avg_improvement = abs_improvement.sum() / bool_updates.sum() if bool_updates.sum() != 0 else None

    nnz_idx = branch_df['best_integer'].to_numpy.nonzero()
    first_incumbent = branch_df['best_integer'].iloc[nnz_idx[0][0]] if len(nnz_idx[0]) != 0 else None

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    second = float(num_updates) / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    sixth = avg_improvement / first_incumbent if avg_improvement and first_incumbent else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive incumbent updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg

    try:
        zeros_stat = [zero_counts.max(), zero_counts.min(), zero_counts.mean(), zeros_to_last]
    except ValueError:
        zeros_stat = [None]*4

    incumbent_list = [
        num_updates,
        second,
        abs_improvement.max(),
        abs_improvement.min(),
        abs_improvement.mean(),
        sixth
    ]
    incumbent_list.extend(zeros_stat)

    if len(incumbent_list) != 10:
        print("***len(incumbent_list): {}".format(len(incumbent_list)))

    return incumbent_list, len(incumbent_list)


def about_best_bound(branch_df):
    """
    number of best_bound updates
    best_bound throughput: num_updates / num_nodes
    max_improvement, min_improvement, avg_improvement
    avg best_bound improvement / first best_bound value
    max, min, avg distance between best_bound updates
    distance between last update and last node explored
    """

    abs_improvement = pd.Series(abs(branch_df['best_bound'].diff(1)))
    bool_updates = pd.Series((abs_improvement != 0))
    avg_improvement = abs_improvement.sum() / bool_updates.sum() if bool_updates.sum() != 0 else None

    nnz_idx = branch_df['best_bound'].to_numpy.nonzero()
    first_best_bound = branch_df['best_bound'].iloc[nnz_idx[0][0]] if len(nnz_idx[0]) != 0 else None

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    second = float(num_updates) / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    sixth = avg_improvement / first_best_bound if avg_improvement and first_best_bound else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive best_bound updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg

    try:
        zeros_stat = [zero_counts.max(), zero_counts.min(), zero_counts.mean(), zeros_to_last]
    except ValueError:
        zeros_stat = [None]*4

    best_bound_list = [
        num_updates,
        second,
        abs_improvement.max(),
        abs_improvement.min(),
        abs_improvement.mean(),
        sixth
    ]
    best_bound_list.extend(zeros_stat)

    if len(best_bound_list) != 10:
        print("***len(best_bound_list): {}".format(len(best_bound_list)))

    return best_bound_list, len(best_bound_list)


def about_objective(branch_df):
    """
    abs(current objective - root objective) (not normalized)
    root objective / current objective
    # nodes with objective 0.05 quantile (0.95) / total number of calls
    last objective / value of 0.05 quantile
    |last best_integer - value of 0.05 quantile|, |last best_bound - value of 0.05 quantile|
    distance between last seen node in 0.05 quantile to end of data collection / total number of calls (rows)
    """
    objective_list = [abs(branch_df['objective'].iloc[-1] - branch_df['objective'].iloc[0])]

    first = branch_df['objective'].iloc[0] / branch_df['objective'].iloc[-1] \
        if branch_df['objective'].iloc[-1] else None
    objective_list.append(first)

    quantile_df = branch_df.loc[branch_df['objective'] > branch_df['objective'].quantile(.95)][['objective',
                                                                                               'times_called']]
    objective_list.append(quantile_df.shape[0] / float(branch_df.times_called.max()))
    second = branch_df.iloc[-1]['objective'] / float(branch_df['objective'].quantile(.95)) \
        if branch_df['objective'].quantile(.95) else None
    objective_list.append(second)
    objective_list.append(abs(branch_df['objective'].quantile(0.95) - branch_df.iloc[-1]['best_integer']))
    objective_list.append(abs(branch_df['objective'].quantile(0.95) - branch_df.iloc[-1]['best_bound']))

    objective_list.append((branch_df.times_called.max() - quantile_df.times_called.max())
                          / float(branch_df.times_called.max()))

    if len(objective_list) != 7:
        print("***len(objective_list): {}".format(len(objective_list)))

    return objective_list, len(objective_list)


def about_active_constraints(branch_df, num_all_constr):
    """
    num_active_constraints / total number of constraints
    num_active_constraints / num_incumbent_active_constraints
    Note: those features were computed only in an older version of the experiments.
    'num_incumbent_active_constraints' is NOT collected anymore in data_run!
    """
    # total_constr = float(miplib_df.loc[miplib_df['Name'] == inst_name]['Rows'])

    # incumbent_active = float(branch_df['num_incumbent_active_constraints'].iloc[-1]) \
    #     if branch_df['num_incumbent_active_constraints'].iloc[-1] else None
    #
    # first = branch_df['num_active_constraints'].iloc[-1]/num_all_constr if num_all_constr != 0 else None
    # second = branch_df['num_active_constraints'].iloc[-1]/incumbent_active if incumbent_active else None
    # active_constr_list = [first, second]
    active_constr_list = [None]*2

    if len(active_constr_list) != 2:
        print("***len(active_constr_list): {}".format(len(active_constr_list)))

    return active_constr_list, len(active_constr_list)


def about_fixed_vars(branch_df, num_all_vars):
    """
    max, min / total # of variables
    last num_fixed_vars / total # of variables
    # nodes with num_fixed_vars in 0.05 quantile (0.95) / total number of calls
    last num_fixed_vars / value of 0.05 quantile
    distance between last seen node in 0.05 quantile to end of data collection / total number of calls (rows)
    """
    fixed_vars_list = [
        branch_df['num_fixed_vars'].max() / float(num_all_vars),
        branch_df['num_fixed_vars'].min() / float(num_all_vars),
        branch_df['num_fixed_vars'].iloc[-1] / float(num_all_vars)
    ]

    quantile_df = branch_df.loc[branch_df['num_fixed_vars'] >
                                branch_df['num_fixed_vars'].quantile(.95)][['num_fixed_vars', 'times_called']]
    fixed_vars_list.append(quantile_df.shape[0] / float(branch_df.times_called.max()))

    first = branch_df['num_fixed_vars'].iloc[-1] / float(branch_df['num_fixed_vars'].quantile(.95)) \
        if branch_df['num_fixed_vars'].quantile(.95) else None
    fixed_vars_list.append(first)

    fixed_vars_list.append((branch_df.times_called.max() - quantile_df.times_called.max())
                           / float(branch_df.times_called.max()))

    if len(fixed_vars_list) != 6:
        print("***len(fixed_vars_list): {}".format(len(fixed_vars_list)))

    return fixed_vars_list, len(fixed_vars_list)


def about_depth(branch_df):
    """
    d_t, max depth of the explored tree
    l_t, last full level of the explored tree
    max_width over levels
    number of levels at max_width
    b_t, waist of the tree / d_t
    b_t, waist of the tree / l_t
    wl_t, waistline (width at level b_t)
    n_t, total number of considered nodes (corresponds to # BranchCB calls)
    """
    d_t = branch_df['depth'].max()
    w_t = branch_df['depth'].value_counts().to_dict()  # {i: width at depth i} for i in [0, d_t]
    gamma_seq = {i: w_t.get(i + 1, 1) / float(w_t.get(i, 1)) for i in range(int(d_t))}  # (def 1 if not found)

    try:
        l_t = np.min([k for k, v in gamma_seq.items() if v < 2])
    except ValueError:
        l_t = 0
    max_width = max(list(w_t.values()))
    max_levels = [key for key, value in w_t.items() if value == max_width]
    b_t = math.ceil((min(max_levels) + max(max_levels)) / 2)
    wl_t = w_t[b_t]
    n_t = np.sum(list(w_t.values()))

    fifth = b_t/float(d_t) if float(d_t) != 0 else None
    sixth = b_t/float(l_t) if float(l_t) != 0 else None

    depth_list = [d_t, l_t, max_width, len(max_levels), fifth, sixth, wl_t, n_t]

    if len(depth_list) != 8:
        print("***len(depth_list): {}".format(len(depth_list)))

    return depth_list, len(depth_list)


def about_dives(branch_df):
    """
    max, min, avg len of dives (jumps)
    # of changes in depth with abs > 1 (identify backtracks between branches and in particular end of dives)
    len of first dive (num_nodes at end of dive) / num_nodes at last seen node
    depth of first dive / max depth in the partial tree
    len of last dive (jump) (num_nodes) / num_nodes at last seen node
    depth of last dive (jump) / max depth in the partial tree
    max, min, avg distance between depth jumps with abs > 1
    distance between last jump and last node explored
    """

    depth_diff = pd.Series(branch_df['depth'].diff(1))

    dives_list = [depth_diff.max(), depth_diff.min(), depth_diff.mean()]

    depth_jump_idx = depth_diff.loc[depth_diff.abs() > 1].index

    dives_list.append(len(depth_jump_idx))

    if len(depth_jump_idx) > 0:
        first_jump_idx = depth_jump_idx[0]
        first_jump_nodes = branch_df.loc[branch_df['global_cb_count'] == first_jump_idx]['num_nodes']
        first_jump_depth = branch_df.loc[branch_df['global_cb_count'] == first_jump_idx]['depth']

        last_jump_idx = depth_jump_idx[-1]
        last_jump_nodes = branch_df.loc[branch_df['global_cb_count'] == last_jump_idx]['num_nodes']
        last_jump_depth = branch_df.loc[branch_df['global_cb_count'] == last_jump_idx]['depth']

        dives_list.append(float(first_jump_nodes) / branch_df.iloc[-1]['num_nodes'])
        dives_list.append(float(first_jump_depth) / branch_df['depth'].max())
        dives_list.append(float(last_jump_nodes) / branch_df.iloc[-1]['num_nodes'])
        dives_list.append(float(last_jump_depth) / branch_df['depth'].max())

        bool_jumps = pd.Series((depth_diff.abs() > 1))
        # add dummy 1 (jump) at the end of bool_jumps
        bool_jumps[bool_jumps.shape[0]] = 1.
        non_zeros = bool_jumps.values == 1
        zeros = ~non_zeros
        zero_counts = np.cumsum(zeros)[non_zeros]
        zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive jumps
        zeros_to_last = zero_counts[-1]
        zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg

        dives_list.extend([zero_counts.max(), zero_counts.min(), zero_counts.mean(), zeros_to_last])
    else:
        dives_list.extend([None]*8)

    if len(dives_list) != 12:
        print("***len(dives_list): {}".format(len(dives_list)))

    return dives_list, len(dives_list)


def about_integral(branch_df, known_opt_value):
    """
    primal integral (using best known solution from miplib_df)
    primal-dual integral (with current best_bound and incumbent)
    """

    total_time = branch_df.iloc[-1]['elapsed']
    # opt = float(miplib_df.loc[miplib_df['Name'] == inst_name]['Objective'])  # best known objective

    # copy part of branch_df
    use_cols = ['elapsed', 'best_integer', 'best_bound']
    copy_branch_df = branch_df[use_cols].copy()
    copy_branch_df['inc_changes'] = abs(copy_branch_df['best_integer'].diff(1))
    copy_branch_df['inc_bool'] = copy_branch_df['inc_changes'] != 0
    copy_branch_df['bb_changes'] = abs(copy_branch_df['best_bound'].diff(1))
    copy_branch_df['bb_bool'] = copy_branch_df['bb_changes'] != 0

    # compute primal integral pi, if value of opt is known
    if known_opt_value:
        primal_dict = OrderedDict()  # {t_i: p(t_i)} for t_i with incumbent change
        primal_dict[0] = 1
        for idx, row in copy_branch_df.loc[copy_branch_df['inc_bool'] != 0].iterrows():
            primal_dict[row['elapsed']] = primal_gap(known_opt_value, row['best_integer'])
        primal_dict[total_time] = None

        times = list(primal_dict.keys())
        integrals = list(primal_dict.values())

        pi = 0
        for i in range(len(times) - 1):
            pi += integrals[i] * (times[i + 1] - times[i])
    else:
        pi = None

    # compute primal-dual integral pdi
    pd_dict = OrderedDict()  # {t_i: pd(t_i)} for t_i with incumbent change
    pd_dict[0] = 1
    for idx, row in copy_branch_df.loc[(copy_branch_df['inc_bool'] != 0) | (copy_branch_df['bb_bool'] != 0)].iterrows():
        pd_dict[row['elapsed']] = primal_dual_gap(row['best_integer'], row['best_bound'])
    pd_dict[total_time] = None

    pd_times = list(pd_dict.keys())
    pd_integrals = list(pd_dict.values())

    pdi = 0
    for i in range(len(pd_times) - 1):
        pdi += pd_integrals[i] * (pd_times[i + 1] - pd_times[i])

    integral_list = [pi, pdi]

    if len(integral_list) != 2:
        print("***len(integral_list): {}".format(len(integral_list)))

    return integral_list, len(integral_list)


def about_cuts(node_df):
    """
    [total number of applied cuts in from_last_node]
    num_cuts applied after root node
    ratio of total / root node cuts
    max, min, avg deltas of cuts additions
    """
    total_cuts = node_df.iloc[-1]['num_cuts']
    root_cuts = node_df.iloc[0]['num_cuts']
    cuts = node_df['num_cuts'].values
    cuts = np.insert(cuts, 0, 0)
    deltas = []
    for i in range(len(cuts) - 1):
        deltas.append(cuts[i + 1] - cuts[i])

    second = total_cuts / float(root_cuts) if float(root_cuts) != 0 else None

    cuts_list = [
        total_cuts - root_cuts,
        second,
        max(deltas),
        min(deltas),
        np.mean(deltas)
    ]

    if len(cuts_list) != 5:
        print("***len(cuts_list): {}".format(len(cuts_list)))

    return cuts_list, len(cuts_list)


def about_open_nodes(node_df, chunk_num, num_nodes_list):
    """
    4x rate of change (slopes) of open_nodes_max in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
    4x rate of change (slopes) of open_nodes_min in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
    4x rate of change (slopes) of num_nodes_at_max in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
    4x rate of change (slopes) of num_nodes_at_min in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
    """
    slopes_max = [None] * 4
    slopes_min = [None] * 4
    slopes_at_max = [None] * 4
    slopes_at_min = [None] * 4

    # Note: old version used node_df as reference for chunks, assuming NodeCB being called at the beginning as well
    # (which does not happen anymore)
    # for i in range(chunk_num):
    #     slopes_max[i] = (node_df['open_nodes_max'].values[i + 1] - node_df['open_nodes_max'].values[i]) \
    #         / num_nodes_list[i] if num_nodes_list[i] != 0 else None
    #     slopes_min[i] = (node_df['open_nodes_min'].values[i + 1] - node_df['open_nodes_min'].values[i]) \
    #         / num_nodes_list[i] if num_nodes_list[i] != 0 else None
    #     slopes_at_max[i] = (node_df['num_nodes_at_max'].values[i + 1] - node_df['num_nodes_at_max'].values[i]) \
    #         / num_nodes_list[i] if num_nodes_list[i] != 0 else None
    #     slopes_at_min[i] = (node_df['num_nodes_at_min'].values[i + 1] - node_df['num_nodes_at_min'].values[i]) \
    #         / num_nodes_list[i] if num_nodes_list[i] != 0 else None

    open_nodes_list = slopes_max + slopes_min + slopes_at_max + slopes_at_min

    if len(open_nodes_list) != 16:
        print("***len(open_nodes_list): {}".format(len(open_nodes_list)))

    return open_nodes_list, len(open_nodes_list)


def about_time(all_df):
    """
    elapsed time
    remaining time
    Note: remaining time not collected anymore!
    """
    # time_list = [all_df['elapsed'].iloc[-1], all_df['remaining_time'].iloc[-1]]
    time_list = [all_df['elapsed'].iloc[-1], None]

    if len(time_list) != 2:
        print("***len(time_list): {}".format(len(time_list)))

    return time_list, len(time_list)


def get_features_label(data_file, data_cols, num_discrete_vars, num_all_vars, num_all_constr, known_opt_value=None):
    """
    :param data_file: file .npz containing data for an (instance, seed) pair
    :param data_cols: column names for data in data_file
    :param num_discrete_vars: number of discrete variables for the problem at hand
    :param num_all_vars: total number of variables
    :param num_all_constr: total number of constraints
    :param known_opt_value: value of optimal solution (if known) else None
    :return: extract features vector for a single data-point from .npz data_file.
    """
    global header_done

    # data_file contains 'ft_matrix', 'label_time', 'label', 'name', 'seed', 'data_final_info' (and others)
    loaded_file = np.load(data_file)
    loaded_data = loaded_file['ft_matrix']

    # check for empty data in is main loop (outside this function)

    # define DataFrame from data
    all_df = pd.DataFrame(loaded_data, columns=data_cols)
    all_df.set_index(all_df['global_cb_count'].values, inplace=True)
    branch_df = all_df.loc[~all_df['index'].isnull()]
    node_df = all_df.loc[all_df['index'].isnull()]

    idx_split, chunk_num, num_nodes_list = get_chunks(node_df, branch_df)

    all_features = list()
    features_cols = OrderedDict()
    # todo: eventually, add here additional columns (e.g., name, seed, dataset)
    all_features.append(float(loaded_file['label_time']))
    all_features.append(float(chunk_num))

    last_data, len_last_data = from_last_seen_data(branch_df.iloc[-1], node_df.iloc[-1])
    all_features.extend(last_data)

    pruned, len_pruned = about_pruned(branch_df)
    all_features.extend(pruned)

    nodes_left, len_nodes_left = about_nodes_left(branch_df, idx_split, num_nodes_list)
    all_features.extend(nodes_left)

    iinf, len_iinf = about_iinf(branch_df, num_discrete_vars)
    all_features.extend(iinf)

    itcnt, len_itcnt = about_itcnt(branch_df)
    all_features.extend(itcnt)

    int_feas, len_int_feas = about_integer_feasible(branch_df)
    all_features.extend(int_feas)

    incumbent, len_incumbent = about_incumbent(branch_df)
    all_features.extend(incumbent)

    best_bound, len_best_bound = about_best_bound(branch_df)
    all_features.extend(best_bound)

    objective, len_objective = about_objective(branch_df)
    all_features.extend(objective)

    active, len_active = about_active_constraints(branch_df, num_all_constr)
    all_features.extend(active)

    fixed, len_fixed = about_fixed_vars(branch_df, num_all_vars)
    all_features.extend(fixed)

    depth, len_depth = about_depth(branch_df)
    all_features.extend(depth)

    dives, len_dives = about_dives(branch_df)
    all_features.extend(dives)

    integral, len_integral = about_integral(branch_df, known_opt_value)
    all_features.extend(integral)

    cuts, len_cuts = about_cuts(node_df)
    all_features.extend(cuts)

    open_nodes, len_open_nodes = about_open_nodes(node_df, chunk_num, num_nodes_list)
    all_features.extend(open_nodes)

    cplex_time, len_time = about_time(all_df)
    all_features.extend(cplex_time)

    # print(
    #     len_last_data, len_pruned, len_nodes_left, len_iinf, len_itcnt, len_int_feas, len_incumbent, len_best_bound,
    #     len_objective, len_active, len_fixed, len_depth, len_dives, len_integral, len_cuts, len_open_nodes, len_time
    # )

    # add label at the end
    all_features.append(float(loaded['label']))

    if not header_done:
        # 'LabelTime' is not added here
        # 'ChunkNum' is not added here
        features_cols['LastData'] = len_last_data
        features_cols['Pruned'] = len_pruned
        features_cols['NodesLeft'] = len_nodes_left
        features_cols['Iinf'] = len_iinf
        features_cols['ItCnt'] = len_itcnt
        features_cols['IntFeas'] = len_int_feas
        features_cols['Incumbent'] = len_incumbent
        features_cols['BestBound'] = len_best_bound
        features_cols['Objective'] = len_objective
        features_cols['ActiveConstr'] = len_active
        features_cols['FixedVars'] = len_fixed
        features_cols['Depth'] = len_depth
        features_cols['Dives'] = len_dives
        features_cols['Integral'] = len_integral
        features_cols['Cuts'] = len_cuts
        features_cols['OpenNodes'] = len_open_nodes
        features_cols['Time'] = len_time
        # 'Label' is not added here

        header_done = True
        print("Len of ft vector: {}".format(len(all_features)))
        return all_features, features_cols

    print("Len of ft vector: {}".format(len(all_features)))
    return all_features


if __name__ == "__main__":

    import sys
    import time
    import os
    import cplex
    import random
    import pickle

    src_dir = os.getcwd()

    """
    Callback data
    """

    COLS = [
        'global_cb_count', 'times_called', 'get_time', 'get_dettime', 'elapsed', 'det_elapsed',
        'index', 'parent_id', 'parent_index',
        'node', 'nodes_left', 'objective', 'iinf', 'best_integer', 'best_bound', 'itCnt', 'gap',
        'num_nodes', 'depth', 'num_fixed_vars', 'is_integer_feasible', 'has_incumbent',
        'open_nodes_len', 'open_nodes_avg', 'open_nodes_min', 'open_nodes_max',
        'num_nodes_at_min', 'num_nodes_at_max', 'num_cuts',
        'cb_time', 'cb_dettime'
    ]

    """
    Dataset split (on names and directly on npz files)
    """

    os.chdir(ARGS.inst_path)

    names_list = list()
    inst_count = 0

    for inst in glob.glob('*.mps.gz'):
        inst_count += 1
        names_list.append(inst)
    print("\nTotal # instances: {}".format(inst_count))

    # remove extension from names
    ext = lambda x: x.rstrip('.mps.gz')
    names_list = [ext(name) for name in names_list]

    # shuffle names and split them
    random.shuffle(names_list)
    # other proportion for train and test split
    train_size = int(math.ceil(3 * len(names_list)) / 5.)  # test_size is 2/5
    list_1 = names_list[:train_size]
    list_2 = names_list[train_size:]
    print("Disjoint lists: {}. Len: {} {}".format(set(list_1).isdisjoint(list_2), len(list_1), len(list_2)))

    os.chdir(src_dir)
    # save list_1 and list_2 for future reference
    with open(os.path.join(ARGS.learning_path, '1_' + ARGS.filename.rstrip('.npz') + '_names.pkl'), 'wb') as f1:
        pickle.dump(list_1, f1)
    f1.close()

    with open(os.path.join(ARGS.learning_path, '2_' +ARGS.filename.rstrip('.npz') + '_names.pkl'), 'wb') as f2:
        pickle.dump(list_2, f2)
    f2.close()

    print("Lengths of splits: {} {}".format(len(list_1), len(list_2)))

    # in subsequent runs, do not perform split but read instead names from loaded lists
    # with open('1_' + ARGS.filename.split('_')[0] + '_names.pkl', 'rb') as f1:
    #     list_1 = pickle.load(f1)
    # f1.close()
    # with open('2_' + ARGS.filename.split('_')[0] + '_names.pkl', 'rb') as f2:
    #     list_2 = pickle.load(f2)
    # f2.close()

    # select file names in npy_path
    os.chdir(ARGS.npy_path)

    npz_1 = [npz for npz in glob.glob('*.npz') if npz.split('_')[0] in list_1]
    npz_2 = [npz for npz in glob.glob('*.npz') if npz.split('_')[0] in list_2]

    # now process one list at a time

    """
    Sets (split) processing
    """

    sets = [npz_1, npz_2]
    set_idx = 0

    for split in sets:
        set_idx += 1
        print("\nProcessing split # {}".format(set_idx))

        os.chdir(ARGS.npy_path)

        count = 0
        count_empty = 0
        glob_list = []
        cols_names = []

        header_done = False  # global

        for f in split:

            count += 1

            # check if data in f is empty
            loaded = np.load(f)  # contains 'ft_matrix', 'label_time', 'label', 'name', 'seed'
            data = loaded['ft_matrix']
            name = str(loaded['name'])  # w/o extension .mps.gz
            print("\n{} {} data type and shape: {} {}".format(count, name, data.dtype, data.shape))

            # check if data is empty
            if data.shape[0] == 0:
                count_empty += 1
                continue  # go to next file

            # read the instance to gather basic variables/constraints info
            os.chdir(ARGS.inst_path)
            c = cplex.Cplex(name + '.mps.gz')
            c.set_results_stream(None)
            c.set_error_stream(None)
            c.set_log_stream(None)
            c.set_warning_stream(None)

            num_discrete = c.variables.get_num_binary() + c.variables.get_num_integer()
            num_vars = c.variables.get_num()
            num_constr = c.linear_constraints.get_num()
            c.end()
            os.chdir(ARGS.npy_path)

            if not header_done:
                # features and header will be returned
                cols_names.append('LabelTime')
                cols_names.append('ChunkNum')
                ft_list, ft_cols = get_features_label(
                    data_file=f,
                    data_cols=COLS,
                    num_discrete_vars=num_discrete,
                    num_all_vars=num_vars,
                    num_all_constr=num_constr,
                    known_opt_value=None
                )
                glob_list.append(np.asarray(ft_list, dtype=np.float))
                for s in ft_cols.keys():
                    cols_names.extend(s + '_' + str(i) for i in range(1, ft_cols[s]+1))
                cols_names.append('Label')
                print("\tHeader created!")
            else:
                glob_list.append(np.asarray(get_features_label(
                    data_file=f,
                    data_cols=COLS,
                    num_discrete_vars=num_discrete,
                    num_all_vars=num_vars,
                    num_all_constr=num_constr,
                    known_opt_value=None
                ),
                    dtype=np.float))

        print("\nCount: {}".format(count))
        print("Empty data count: {}".format(count_empty))
        print("Len of glob_list: {}".format(len(glob_list)))
        global_arr = np.asarray(glob_list, dtype=np.float)
        print("Shape of global_arr: {}".format(global_arr.shape))
        print("\n{}".format(cols_names))

        # print("\nSample: ")
        # print(global_arr[0])

        os.chdir(ARGS.learning_path)
        np.savez(str(set_idx) + '_' + ARGS.filename, data=global_arr, cols=cols_names)
