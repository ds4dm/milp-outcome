import os
import argparse
import pandas as pd
import numpy as np
import glob
import math
from collections import OrderedDict

from pandas.io.common import EmptyDataError

import shared
import utilities


"""

TASK 1: 
    Iterate through different rhos, and through .stat files, to compute label, node_limit and trivial flag 
    with respect to different time limits (taus).
    
    Outputs for the same rho are saved in /RUNS_INFO/ as rho_dataset_seed.txt files:
        name, seed, rho, tl, tau, nodes_tau, gap_tau, b_calls_tau, label, trivial, 
        gap, sol_time, nodes, b_calls, status
    that can be later loaded as DataFrame for analysis, 
    and is also used to automatically generate commands for data_run (utilities.write_data_jobs_args).
    
TASK 2:
    Iterate through .stat files, and through different rhos, to create and pickle name_seed.pkl dictionaries
    in TIME_NODES_DICT/dataset_seed/.
    Dictionaries store the number of nodes at 25, 50, 75, 100 % of tau, 
    which will be used as marks for NodeCB in data_run.
    
Run as
>>> python read_stats.py --seed 201610271 --dataset Benchmark78

"""


""" 
Argparser definition 
"""

parser = argparse.ArgumentParser(description='Arg parser for read_stats script.')

# Paths
parser.add_argument(
    '--stat_path',
    type=str,
    default=shared.DATA_PATH + '/STAT/',
    help='Absolute path to stat archive.'
)
parser.add_argument(
    '--data_path',
    type=str,
    default=shared.DATA_PATH,
    help='Absolute path to data archive. Can be set within shared.py.'
)

# Specifications
parser.add_argument(
    '--seed',
    type=int,
    default=201610271,
    help='Sets the solver random seed. Default is 201610271 (CPLEX version 12.7.1).'
)
parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    help='Dataset of origin identifier for the MILP instance (e.g., Benchmark78).'
)
parser.add_argument(
    '--tls',
    type=float,
    nargs='+',
    default=[1200., 2400., 3600., 7200.],
    help='List of TLs for which time stamps are gathered.'
)
parser.add_argument(
    '--rhos',
    type=float,
    nargs='+',
    default=[5, 10, 15, 20, 25],
    help='List of RHOs for which time stamps are gathered.'
)

ARGS = parser.parse_args()


"""
Time limits and rho-stamps definition.
"""

TLs = ARGS.tls
RHOs = ARGS.rhos

TL_dict = {}
for tl in TLs:
    TL_dict[tl] = [rho * tl/100 for rho in RHOs]

RHO_dict = {}
for rho in RHOs:
    RHO_dict[rho] = [rho * tl/100 for tl in TLs]

STAMPS = [rho * tl/100 for rho in RHOs for tl in TLs]
STAMPS.extend(TLs)
STAMPS = list(set(STAMPS))  # remove duplicates and sort
STAMPS.sort()

ALL_STAMPS = []
for k in STAMPS:
    ALL_STAMPS.extend([k/100.*p for p in [25, 50, 75, 100]])

ALL_STAMPS_U = list(set(ALL_STAMPS))
ALL_STAMPS_U.sort()

ALL_STAMPS_flags = OrderedDict()
for t_stamp in ALL_STAMPS_U:
    ALL_STAMPS_flags[t_stamp] = False

# rho_tau_nodes data:
# given rho, for all tau collects the # of nodes processed at {25, 50, 75}% of tau
# nodes stamps will be used in data_run as markers for NodeCB

RHO_TAU_NODES = OrderedDict()
for rho in RHOs:
    RHO_TAU_NODES[rho] = OrderedDict((tau, []) for tau in RHO_dict[rho])


if __name__ == "__main__":

    import pickle

    dir_info_str = ARGS.dataset + '_' + str(ARGS.seed) + '/'
    stat_dir = ARGS.data_path + "/STAT/" + dir_info_str

    time_nodes_dir = ARGS.data_path + "/TIME_NODES_DICT/" + dir_info_str

    try:
        os.mkdir(time_nodes_dir)
    except OSError:
        if not os.path.isdir(time_nodes_dir):
            raise

    os.chdir(stat_dir)

    # columns of the .stat files
    cols = [
        "NAME", "SEED", "TIME_STAMP", "NODES", "ITCNT", "GAP",
        "ELAPSED_SECS", "ELAPSED_TICKS", "B_CALLS",
        "BEST_BOUND", "INCUMBENT", "OBJECTIVE",
        "STATUS", "SOL_TIME", "END_LINE"
    ]

    nan_dict = {
        "TIME_STAMP": "None", "NODES": "None", "ITCNT": "None", "GAP": "None",
        "ELAPSED_SECS": "None", "ELAPSED_TICKS": "None",
        "BEST_BOUND": "None", "INCUMBENT": "None", "OBJECTIVE": "None",
        "STATUS": "None", "SOL_TIME": "None", "END_LINE": "False"
    }

    # header of rho_dataset_seed.txt info file
    header = [
        "NAME", "SEED", "RHO", "TL", "TAU", "NODES_TAU", "GAP_TAU", "B_CALLS_TAU", "LABEL", "TRIVIAL",
        "GAP", "SOL_TIME", "NODES", "B_CALLS", "STATUS"
    ]

    #####################################
    # NOTE:
    # the only discarded instances are those raising EmptyDataError.
    # All the other runs (.stat files) are processed and outputs are created for them.
    # They can be excluded from data collection in subsequent steps. (e.g., by selecting part of the
    # rho_dataset_seed.txt data to generate commands for data_run.py (in utilities)
    #####################################

    #####################################
    # TASK 1:
    # iterate through rho in RHOs, and through stat_file in stat_dir to
    # create rho_dataset_seed.txt in RUNS_INFO
    #####################################

    print("\n\n################### CREATE RUNS_INFO ###################")

    file_count_1 = 0
    count_error_1 = 0

    for rho in RHOs:
        print("\nProcessing rho = {}".format(rho))
        file_name = str(int(rho)) + '_' + ARGS.dataset + '_' + str(ARGS.seed)
        info_file = os.path.join(ARGS.data_path + "/RUNS_INFO/", file_name + '.txt')

        count_single_row_solved = 0
        count_not_mapped = 0
        count_single_not_solved = 0
        count_empty_data = 0

        with open(info_file, 'a') as info_append:
            for item in header:
                info_append.write("%s\t" % item)
            info_append.write("\n")

            for stat_file in glob.glob('*.stat'):
                # print(stat_file)
                file_count_1 += 1
                try:
                    df = pd.read_csv(stat_file, usecols=cols, header=0, sep='\t',
                                     na_values=nan_dict, keep_default_na=False, false_values="False")
                except EmptyDataError:
                    print(" !!! EmptyDataError on file: {}".format(stat_file))
                    count_empty_data += 1
                    continue

                df['TIME_STAMP'] = df['TIME_STAMP'].apply(pd.to_numeric, errors='coerce')
                df[['BEST_BOUND', 'INCUMBENT']] = df[['BEST_BOUND', 'INCUMBENT']].apply(pd.to_numeric, errors='coerce')
                df['STATUS'] = df['STATUS'].apply(pd.to_numeric, errors='coerce')
                # print(df.dtypes)

                # identify last row in stat file
                last = df.tail(1)
                name = last['NAME'].values[0]
                seed = last['SEED'].values[0]
                inst_info_str = name + '_' + str(seed)

                tot_time = last['SOL_TIME'].values[0]  # corresponds to solution time, IF problem solved to optimality
                status = last['STATUS'].values[0]
                tot_nodes = last['NODES'].values[0]
                tot_b_calls = last['B_CALLS'].values[0]

                if math.isnan(df.iloc[-1]['SOL_TIME']):
                    print("\t{} Error raised: {}. Shape: {}".format(name, last['STATUS'].values[0], df.shape))
                    count_error_1 += 1  # comprises errors in single row as well
                    # NOTE: error instances are not discarded upfront.

                # select (tau, tl) pairs depending on rho
                taus = RHO_dict[rho]
                # print("Tau stamps for rho = {} are {}".format(rho, taus))

                # case 0: stat file contains only end_line
                # need to differentiate depending on status
                if (df.shape[0] == 1) and (status in [101, 102]):
                    # NOTE: in this case sol_time < smallest stamp,
                    # so label (=1) and trivial flag (=True) will be the same for all (tau, tl)
                    # with default parameters smallest stamp is 15 secs (i.e., 25% of 60 secs)
                    print("\t{} Shape of data is (1, -) with status {}".format(name,  df.iloc[-1]['STATUS']))
                    count_single_row_solved += 1
                    for tau in taus:
                        count_not_mapped += 1
                        tl = tau / rho * 100
                        # print(" ..processing (tau, tl): {} {}".format(tau, tl))
                        #  "NAME", "SEED", "RHO", "TL", "TAU", "NODES_TAU", "GAP_TAU", "B_CALLS_TAU",
                        #  "LABEL", "TRIVIAL",
                        #  "GAP", "SOL_TIME", "NODES", "B_CALLS", "STATUS"

                        # NOTE: value of nodes, gap, b_calls at tau correspond with total ones (tau was not hit)
                        line = [
                            name, seed, rho, tl, tau, tot_nodes, last['GAP'].values[0], tot_b_calls, 1, True,
                            last['GAP'].values[0], tot_time, tot_nodes, tot_b_calls, status
                        ]

                        for entry in line:
                            info_append.write("%s\t" % entry)
                        info_append.write("\n")

                elif (df.shape[0] == 1) and not (status in [101, 102]):
                    # case of instances stuck (at root), label and trivial flag will be 0 and False
                    print("\t{} Shape of data is (1, -) with status {}".format(name, df.iloc[-1]['STATUS']))
                    count_single_not_solved += 1
                    for tau in taus:
                        count_not_mapped += 1
                        tl = tau / rho * 100
                        # print(" ..processing (tau, tl): {} {}".format(tau, tl))
                        #  "NAME", "SEED", "RHO", "TL", "TAU", "NODES_TAU", "GAP_TAU", "B_CALLS_TAU",
                        #  "LABEL", "TRIVIAL",
                        #  "GAP", "SOL_TIME", "NODES", "B_CALLS", "STATUS"

                        # NOTE: value of nodes, gap, b_calls at tau correspond with total ones (tau was not hit)
                        line = [
                            name, seed, rho, tl, tau, tot_nodes, last['GAP'].values[0], tot_b_calls, 0, False,
                            last['GAP'].values[0], tot_time, tot_nodes, tot_b_calls, status
                        ]

                        for entry in line:
                            info_append.write("%s\t" % entry)
                        info_append.write("\n")

                # case 1: stat file contains multiple lines
                elif df.shape[0] > 1:
                    print("\t{}".format(name))
                    mapped = [k for k in df['TIME_STAMP'].values if not np.isnan(k)]  # time stamps successfully mapped
                    for tau in taus:
                        tl = tau / rho * 100
                        # print(" ..processing (tau, tl): {} {}".format(tau, tl))
                        # process taus that were mapped during the run and determine label and trivial flag
                        if tau in mapped:
                            tau_row = df.loc[df['TIME_STAMP'] == tau]
                            label = 1 if tot_time <= tl else 0
                            trivial = True if tot_time <= tau else False

                            # assertion test (tot_time might be fractions of seconds bigger than tl)
                            if label == 0:
                                assert status == 107

                            #  "NAME", "SEED", "RHO", "TL", "TAU", "NODES_TAU", "GAP_TAU", "B_CALLS_TAU",
                            #  "LABEL", "TRIVIAL",
                            #  "GAP", "SOL_TIME", "NODES", "B_CALLS", "STATUS"

                            line = [
                                name,
                                seed,
                                rho,
                                tl,
                                tau,
                                tau_row['NODES'].values[0],
                                tau_row['GAP'].values[0],
                                tau_row['B_CALLS'].values[0],
                                label,
                                trivial,
                                last['GAP'].values[0],
                                tot_time,
                                tot_nodes,
                                tot_b_calls,
                                status
                            ]
                            for entry in line:
                                info_append.write("%s\t" % entry)
                            info_append.write("\n")

                        # process taus that were not mapped
                        # here sol_time < tau, so label and trivial flag will be at 1 and True
                        # and nodes, gap, b_calls at tau correspond to total ones (tau was not hit)
                        else:
                            count_not_mapped += 1
                            line = [
                                name, seed, rho, tl, tau, tot_nodes, last['GAP'].values[0], tot_b_calls, 1, True,
                                last['GAP'].values[0], tot_time, tot_nodes, tot_b_calls, status
                            ]
                            for entry in line:
                                info_append.write("%s\t" % entry)
                            info_append.write("\n")
                else:
                    print("\t{} Shape of stat DataFrame is {}. Discarded.".format(name, df.shape[0]))
                    count_error_1 += 1

        info_append.close()

        print("\nSingle row files: {}".format(count_single_row_solved))
        print("Single row files not solved: {}".format(count_single_not_solved))
        print("Total cases of non-mapped tau: {}".format(count_not_mapped))
        print("Empty data cases: {}".format(count_empty_data))

    print("\nTotal processed stat files: {} * {} ({})".format(int(file_count_1/float(len(RHOs))), len(RHOs), file_count_1))
    print("Errors instances: {}".format(count_error_1))

    #####################################
    # TASK 2:
    # iterate through stat_file in stat_dir, and through rho in RHOs to
    # create and pickle name_seed {} in TIME_NODES_DICT/dataset_seed/
    #####################################

    print("\n\n################### CREATE TIME_NODES_DICT ###################")

    file_count_2 = 0
    count_error_2 = 0
    count_empty_data = 0

    for stat_file in glob.glob('*.stat'):
        # print("\n{}".format(stat_file))
        file_count_2 += 1
        try:
            df = pd.read_csv(stat_file, usecols=cols, header=0, sep='\t',
                             na_values=nan_dict, keep_default_na=False, false_values="False")
        except EmptyDataError:
            print(" !!! EmptyDataError on file: {}".format(stat_file))
            count_empty_data += 1
            continue

        df['TIME_STAMP'] = df['TIME_STAMP'].apply(pd.to_numeric, errors='coerce')
        df['STATUS'] = df['STATUS'].apply(pd.to_numeric, errors='coerce')

        # identify last row in stat file
        last = df.tail(1)
        name = last['NAME'].values[0]
        seed = last['SEED'].values[0]
        inst_info_str = name + '_' + str(seed)

        if math.isnan(df.iloc[-1]['SOL_TIME']):
            print("\t{} Error raised: {}. Shape: {}".format(name, last['STATUS'].values[0], df.shape))
            count_error_2 += 1  # comprises errors in single row as well
            # NOTE: error instances are not discarded upfront.

        print("\n{}".format(name))
        for rho in RHOs:
            print("\trho = {}".format(rho))

            count_single_row_solved = 0
            count_not_mapped = 0
            count_single_not_solved = 0

            # select (tau, tl) pairs depending on rho
            taus = RHO_dict[rho]
            # print("Tau stamps for rho = {} are {}".format(rho, taus))

            # case 0: stat file contains only end_line
            # need to differentiate depending on status
            if (df.shape[0] == 1) and (last['STATUS'].values[0] in [101, 102]):
                print("\t\t{} Shape of data is (1, -) with status {}".format(name, df.iloc[-1]['STATUS']))
                count_single_row_solved += 1
                for tau in taus:
                    count_not_mapped += 1
                    # fill RHO_TAU_NODES[rho][tau] with None, because even the smallest stamp was not recorded
                    RHO_TAU_NODES[rho][tau] = [None] * 4

            elif (df.shape[0] == 1) and not (last['STATUS'].values[0] in [101, 102]):
                # case of instances stuck at root, label and trivial flag will be 0 and False
                print("\t\t{} Shape of data is (1, -) with status {}".format(name, df.iloc[-1]['STATUS']))
                count_single_not_solved += 1
                for tau in taus:
                    count_not_mapped += 1
                    # fill RHO_TAU_NODES[rho][tau] with None, because even the smallest stamp was not recorded
                    RHO_TAU_NODES[rho][tau] = [None] * 4

            # case 1: stat file contains multiple lines (i.e., some stamps were recorded)
            elif df.shape[0] > 1:
                mapped = [k for k in df['TIME_STAMP'].values if not np.isnan(k)]  # time stamps mapped
                for tau in taus:
                    tl = tau / rho * 100
                    # process taus that were mapped during the run
                    if tau in mapped:
                        # fill RHO_TAU_NODES[rho][tau]
                        tau_p = [tau / 100. * p for p in [25, 50, 75, 100]]  # all are mapped
                        RHO_TAU_NODES[rho][tau] = [df.loc[df['TIME_STAMP'] == tp]['NODES'].values[0] for tp in tau_p]

                    # process taus that were not mapped
                    else:
                        count_not_mapped += 1
                        # fill RHO_TAU_NODES[rho][tau]
                        tau_p = [tau / 100. * p for p in [25, 50, 75, 100]]
                        dict_list = []
                        for tp in tau_p:
                            if tp in mapped:
                                dict_list.append(df.loc[df['TIME_STAMP'] == tp]['NODES'].values[0])
                            else:
                                dict_list.append(None)
                        assert len(dict_list) == 4
                        RHO_TAU_NODES[rho][tau] = dict_list
            else:
                print("\t\t{} Shape of stat DataFrame is {}. Discarded.".format(name, df.shape[0]))
                count_error_2 += 1

            print("\t\tSingle row files: {}".format(count_single_row_solved))
            print("\t\tSingle row files: {}".format(count_single_not_solved))
            print("\t\tTotal cases of non-mapped tau: {}".format(count_not_mapped))

        # save dictionary for the instance
        with open(os.path.join(time_nodes_dir, inst_info_str + '.pkl'), 'wb') as tn_f:
            pickle.dump(RHO_TAU_NODES, tn_f, pickle.DEFAULT_PROTOCOL)
        tn_f.close()

    print("\nTotal processed stat files: {}".format(file_count_2))
    print("Errors instances: {}".format(count_error_2))
    print("Empty data cases: {}".format(count_empty_data))
