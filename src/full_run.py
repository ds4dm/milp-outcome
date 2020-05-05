import argparse
from collections import OrderedDict

import cplex
from cplex.callbacks import BranchCallback
from cplex.callbacks import NodeCallback
from cplex.exceptions import CplexSolverError

import shared
import utilities

""" 
Run (instance, seed) with single (longest specified) time limit (2h).
Collect basic info on the run at predefined time stamps (tau, depending on tl and rho).
Note: these are not the basic info that are used to build features, but will be used to define subsequent runs.

It is not possible to control the invocation of MIPInfoCallback, hence
basic stats on the run are collected via BranchCallback.

Run as
>>> python full_run.py --instance air04.mps.gz --dataset Benchmark78

Default random seed:
- 12.8.0.0 : 201709013
- 12.7.1.0 : 201610271

"""


""" 
Argparser definition.
"""

parser = argparse.ArgumentParser(description='Arg parser for full_run script.')

# Run identifiers: instance name, random seed, dataset
parser.add_argument(
    '--instance',
    type=str,
    required=True,
    help='Name of the MIP instance (extension included, e.g., air04.mps.gz).'
)
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

# Node and time limit settings:
parser.add_argument(
    '--time_limit',
    type=float,
    default=7200,
    help='Sets the time limit (secs) for the full run. Default is 7200.'
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

# Paths
parser.add_argument(
    '--data_path',
    type=str,
    default=shared.DATA_PATH,
    help='Absolute path to data archive. Can be set in shared.py.'
)
parser.add_argument(
    '--inst_path',
    type=str,
    default=shared.INST_PATH,
    help='Absolute path to instances. Can be set in shared.py.'
)

ARGS = parser.parse_args()


"""
Time limits and rho-stamps definition.
"""

# TLs = [1200., 2400., 3600., 7200.]
# RHOs = [5, 10, 15, 20, 25]

# TLs = [1500., 6000.]
# RHOs = [20]

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
for k in STAMPS:  # TLs are also considered
    ALL_STAMPS.extend([k/100.*p for p in [25, 50, 75, 100]])

ALL_STAMPS_U = list(set(ALL_STAMPS))
ALL_STAMPS_U.sort()

ALL_STAMPS_flags = OrderedDict()
for t_stamp in ALL_STAMPS_U:
    ALL_STAMPS_flags[t_stamp] = False


""" 
Callbacks definition.
"""


class MyEmptyNode(NodeCallback):
    """
    Empty callback. Custom subclass of NodeCallback.
    This callback will be used *before CPLEX enters a node*.
    """
    def __init__(self, env):
        NodeCallback.__init__(self, env)
        self.times_called = 0

    def __call__(self):
        self.times_called += 1


class MyStatBranch(BranchCallback):
    """
    Custom subclass of BranchCallback.
    Tracks basic data on the optimization process, at predetermined time intervals.
    Data is written to globally open file stats_append.
    This callback will be used *prior to branching* at a node in the branch and cut tree.
    """
    def __init__(self, env):
        BranchCallback.__init__(self, env)
        self.times_called = 0

    def __call__(self):
        self.times_called += 1
        global ARGS
        global stats_append
        global ALL_STAMPS_flags

        elapsed = self.get_time() - self.get_start_time()
        elapsed_ticks = self.get_dettime() - self.get_start_dettime()
        # measure overhead of the whole CB call
        # cb_start = self.get_time()

        # save stats according to time stamps
        for stamp in ALL_STAMPS_flags.keys():
            if elapsed >= stamp and not ALL_STAMPS_flags[stamp]:
                ALL_STAMPS_flags[stamp] = True  # might not be hit if stamp == timelimit
                #  "NAME", "SEED", "TIME_STAMP", "NODES", "ITCNT", "GAP",
                #  "ELAPSED_SECS", "ELAPSED_TICKS", "B_CALLS",
                #  "BEST_BOUND", "INCUMBENT", "OBJECTIVE",
                #  "STATUS", "SOL_TIME", "END_LINE"
                name = ARGS.instance.split('.')[0]
                line = [name, ARGS.seed, stamp, self.get_num_nodes(),
                        self.get_num_iterations(), self.get_MIP_relative_gap(), elapsed, elapsed_ticks,
                        self.times_called, self.get_best_objective_value(), self.get_incumbent_objective_value(),
                        self.get_objective_value(), None, None, False]
                # "STATUS" and "SOL_TIME" are filled in after solve() call is over, with "END_LINE" True (in the main)
                for entry in line:
                    stats_append.write("%s\t" % entry)
                stats_append.write("\n")
                break
        # cb_time = self.get_time() - cb_start
        # print("cb_time: {}".format(cb_time))


if __name__ == "__main__":

    import sys
    import os.path

    cwd = os.getcwd()

    name = ARGS.instance.split('.')[0]
    inst_info_str = name + '_' + str(ARGS.seed)
    dir_info_str = ARGS.dataset + '_' + str(ARGS.seed) + '/'

    # data directory setup
    utilities.dir_setup(parent_path_inst_dir=ARGS.data_path, rhos=ARGS.rhos)

    try:
        os.mkdir(ARGS.data_path + "/OUT/" + dir_info_str)
    except OSError:
        if not os.path.isdir(ARGS.data_path + "/OUT/" + dir_info_str):
            raise

    os.chdir(ARGS.data_path + "/OUT/" + dir_info_str)
    sys.stdout = open(inst_info_str + '.out', 'w')  # output file

    os.chdir(ARGS.inst_path)
    pb = cplex.Cplex(ARGS.instance)

    print("\nName: {}".format(pb.get_problem_name()))
    print("Type: {}".format(pb.problem_type[pb.get_problem_type()]))
    print("# Variables: {}".format(pb.variables.get_num()))
    print("# Constraints: {}".format(pb.linear_constraints.get_num()))
    print("Seed: {}".format(ARGS.seed))

    # set parameters
    utilities.set_solver_full_run_parameters(pb, inst_info_str, dir_info_str, ARGS)

    # register callback
    branch_instance = pb.register_callback(MyStatBranch)
    node_instance = pb.register_callback(MyEmptyNode)

    header = [
        "NAME", "SEED", "TIME_STAMP", "NODES", "ITCNT", "GAP",
        "ELAPSED_SECS", "ELAPSED_TICKS", "B_CALLS",
        "BEST_BOUND", "INCUMBENT", "OBJECTIVE",
        "STATUS", "SOL_TIME", "END_LINE"
    ]

    try:
        os.mkdir(ARGS.data_path + "/STAT/" + dir_info_str)
    except OSError:
        if not os.path.isdir(ARGS.data_path + "/STAT/" + dir_info_str):
            raise

    with open(os.path.join(ARGS.data_path + "/STAT/" + dir_info_str, inst_info_str + '.stat'), 'a') as stats_append:
        for item in header:
            stats_append.write("%s\t" % item)
        stats_append.write("\n")

        try:
            t0, det_t0 = pb.get_time(), pb.get_dettime()
            print("\nInitial time-stamps: {} {}".format(t0, det_t0))
            pb.solve()
            sol_time_secs = pb.get_time() - t0
            sol_time_ticks = pb.get_dettime() - det_t0
            print("\nFinal elapsed: {} {}".format(sol_time_secs, sol_time_ticks))

            # termination line
            end_line = [
                name,
                ARGS.seed,
                None,
                int(pb.solution.progress.get_num_nodes_processed()),
                int(pb.solution.progress.get_num_iterations()),
                pb.solution.MIP.get_mip_relative_gap(),
                sol_time_secs,
                sol_time_ticks,
                branch_instance.times_called,
                pb.solution.MIP.get_best_objective(),
                None,
                None,
                pb.solution.get_status(),
                sol_time_secs,
                True
            ]

            for item in end_line:
                stats_append.write("%s\t" % item)
            stats_append.write("\n")

            print("\nStatus: {}".format(pb.solution.get_status()))

        except CplexSolverError as exc:  # end-line data will be all zeros if exception is raised
            print("Exception raised during solve")
            end_line = [
                name, ARGS.seed, None, None, None, None,
                None, None, branch_instance.times_called,
                None, None, None,
                exc.args[2], None, True
            ]

            for item in end_line:
                stats_append.write("%s\t" % item)
            stats_append.write("\n")

        print("\nMyStatBranch # calls: {}".format(branch_instance.times_called))
        print("MyEmptyNode # calls: {}".format(node_instance.times_called))
        print("\nMyStatBranch flags: ")
        print(ALL_STAMPS_flags)

        pb.end()
    stats_append.close()
