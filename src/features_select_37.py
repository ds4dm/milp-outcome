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

In this script, we just consider the 37 features that were selected for learning in the CPAIOR 2019 paper.
For the complete feature set see features_all.py and the additional ones in learning.ipynb.

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
    "data_final_info: array of final information on the run (useful to check trivial instances!)
    
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
Selected features (37)
"""


def get_37_features(branch_df, node_df, num_discrete_vars, num_all_vars):
    """
    :param branch_df:
    :param node_df:
    :param num_discrete_vars:
    :param num_all_vars:
    :return:
    """
    last_branch = branch_df.iloc[-1]
    last_node = node_df.iloc[-1]

    fts_list = []

    # from last seen data (6)
    ls7 = last_branch['gap']
    ls12 = last_branch['best_bound'] / last_branch['best_integer'] \
        if last_branch['best_integer'] and last_branch['best_bound'] else None
    ls17 = last_node['num_nodes_at_max'] / float(last_node['open_nodes_len'])
    ls18 = last_node['num_nodes_at_min'] / float(last_node['open_nodes_len'])
    ls19 = last_node['open_nodes_max'] / float(last_branch['best_integer']) \
        if last_branch['best_integer'] and last_node['open_nodes_max'] else None
    ls20 = last_node['open_nodes_min'] / float(last_branch['best_integer']) \
        if last_branch['best_integer'] and last_node['open_nodes_min'] else None

    fts_list.extend([ls7, ls12, ls17, ls18, ls19, ls20])

    # about pruned (2)
    pruned = pd.Series(branch_df['num_nodes'].diff(1) - 1)  # correction!
    cumulative = float(pruned.sum())
    p2 = cumulative / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    p3 = cumulative / branch_df['nodes_left'].iloc[-1] if branch_df['nodes_left'].iloc[-1] != 0 else None

    fts_list.extend([p2, p3])

    # about nodes left (1)
    left2 = branch_df.iloc[-1]['nodes_left'] / branch_df['nodes_left'].max()

    fts_list.extend([left2])

    # about iinf (4)
    quantile_df = branch_df.loc[branch_df['iinf'] < branch_df['iinf'].quantile(.05)][['iinf', 'times_called']]
    iinf1 = branch_df['iinf'].max() / float(num_discrete_vars)
    iinf2 = branch_df['iinf'].min() / float(num_discrete_vars)
    iinf3 = branch_df['iinf'].mean() / float(num_discrete_vars)
    iinf5 = quantile_df.shape[0] / float(branch_df.times_called.max())

    fts_list.extend([iinf1, iinf2, iinf3, iinf5])

    # about itcnt (1)
    itcnt1 = branch_df['itCnt'].iloc[-1] / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 \
        else None

    fts_list.extend([itcnt1])

    # about integer feasible (1)
    if branch_df.loc[branch_df['has_incumbent'] == 1].shape[0] == 0:
        feas2 = False
    else:
        first_inc_row = branch_df.loc[branch_df['has_incumbent'] == 1][
            ['global_cb_count', 'times_called', 'is_integer_feasible', 'has_incumbent', 'depth']].iloc[0]
        feas2 = True if \
            (not first_inc_row['is_integer_feasible']) & (first_inc_row['has_incumbent']) \
            else False

    fts_list.extend([feas2])

    # needed depth data
    w_t = branch_df['depth'].value_counts().to_dict()  # {i: width at depth i} for i in [0, d_t]
    n_t = np.sum(list(w_t.values()))

    # about incumbent (4)
    abs_improvement = pd.Series(abs(branch_df['best_integer'].diff(1)))
    bool_updates = pd.Series((abs_improvement != 0))

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    inc2 = float(num_updates) / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    incA5 = abs_improvement.mean() / last_branch['best_integer'] if last_branch['best_integer'] else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive incumbent updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg
    try:
        inc9 = zero_counts.mean()
        inc10 = zeros_to_last
    except ValueError:
        inc9 = None
        inc10 = None
    incA9 = inc9 / n_t if inc9 and n_t else None
    incA10 = inc10 / inc9 if inc9 else None

    fts_list.extend([inc2, incA5, incA9, incA10])

    # about best bound (4)
    abs_improvement = pd.Series(abs(branch_df['best_bound'].diff(1)))
    bool_updates = pd.Series((abs_improvement != 0))
    avg_improvement = abs_improvement.sum() / bool_updates.sum() if bool_updates.sum() != 0 else None

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    bb2 = float(num_updates) / branch_df['num_nodes'].iloc[-1] if branch_df['num_nodes'].iloc[-1] != 0 else None
    bbA5 = abs_improvement.mean() / last_branch['best_bound'] if last_branch['best_bound'] else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive best_bound updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg
    try:
        bb9 = zero_counts.mean()
        bb10 = zeros_to_last
    except ValueError:
        bb9 = None
        bb10 = None
    bbA9 = bb9 / n_t if bb9 and n_t else None
    bbA10 = bb10 / bb9 if bb9 else None

    fts_list.extend([bb2, bbA5, bbA9, bbA10])

    # about objective (3)
    quantile_df = branch_df.loc[branch_df['objective'] > branch_df['objective'].quantile(.95)][['objective',
                                                                                                'times_called']]
    obj3 = quantile_df.shape[0] / float(branch_df.times_called.max())
    obj5 = abs(branch_df['objective'].quantile(0.95) - branch_df.iloc[-1]['best_integer'])
    obj6 = abs(branch_df['objective'].quantile(0.95) - branch_df.iloc[-1]['best_bound'])

    fts_list.extend([obj3, obj5, obj6])

    # about fixed variables (4)
    fix1 = branch_df['num_fixed_vars'].max() / float(num_all_vars)
    fix2 = branch_df['num_fixed_vars'].min() / float(num_all_vars)
    quantile_df = branch_df.loc[branch_df['num_fixed_vars'] >
                                branch_df['num_fixed_vars'].quantile(.95)][['num_fixed_vars', 'times_called']]
    fix4 = quantile_df.shape[0] / float(branch_df.times_called.max())
    fix6 = (branch_df.times_called.max() - quantile_df.times_called.max()) / float(branch_df.times_called.max())

    fts_list.extend([fix1, fix2, fix4, fix6])

    # about depth (3)
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

    depthA1 = d_t / last_branch['num_nodes']
    depthA2 = l_t / d_t
    depth5 = b_t/float(d_t) if float(d_t) != 0 else None

    fts_list.extend([depthA1, depthA2, depth5])

    # about dives (3)
    depth_diff = pd.Series(branch_df['depth'].diff(1))
    dive1 = depth_diff.max()
    dive3 = depth_diff.mean()
    depth_jump_idx = depth_diff.loc[depth_diff.abs() > 1].index
    diveA4 = len(depth_jump_idx) / n_t

    fts_list.extend([dive1, dive3, diveA4])

    # about integral (1)
    total_time = branch_df.iloc[-1]['elapsed']
    # opt = float(miplib_df.loc[miplib_df['Name'] == inst_name]['Objective'])  # best known objective

    # copy part of branch_df
    use_cols = ['elapsed', 'best_integer', 'best_bound']
    copy_branch_df = branch_df[use_cols].copy()
    copy_branch_df['inc_changes'] = abs(copy_branch_df['best_integer'].diff(1))
    copy_branch_df['inc_bool'] = copy_branch_df['inc_changes'] != 0
    copy_branch_df['bb_changes'] = abs(copy_branch_df['best_bound'].diff(1))
    copy_branch_df['bb_bool'] = copy_branch_df['bb_changes'] != 0

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

    fts_list.extend([pdi])

    return fts_list


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

    all_features = get_37_features(branch_df=branch_df, node_df=node_df,
                                   num_discrete_vars=num_discrete_vars, num_all_vars=num_all_vars)
    all_features.insert(0, float(loaded_file['label_time']))
    all_features.insert(1, float(chunk_num))
    all_features.append(float(loaded['label']))

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

    for inst in glob.glob('*.mps.gz'):  # extension dependency
        inst_count += 1
        names_list.append(inst)
    print("\nTotal # instances: {}".format(inst_count))

    # remove extension from names
    ext = lambda x: x.rstrip('.mps.gz')  # extension dependency
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

        # print("\nSample: ")
        # print(global_arr[0])

        os.chdir(ARGS.learning_path)
        np.savez(str(set_idx) + '_37_' + ARGS.filename, data=global_arr)
