import numpy as np
import pandas as pd
import os
import cplex
from cplex.exceptions import CplexSolverError
from cplex.callbacks import BranchCallback
from cplex.callbacks import NodeCallback

import pickle
import argparse
from collections import OrderedDict

import shared
import utilities


"""
Run as
>>> python data_run.py --instance air04.mps.gz --seed 20180329 --node_limit $NODELIM

Extract information from the B&B tree via BranchCB and NodeCB.

Callbacks are invoked by the solver at *every* branched node, and at each node selection, but we extract information 
only up to a certain number of nodes (--node_limit), before stopping the run. 
Node limit is computed from a full run, as the # of nodes processed up to rho% of a certain time_limit 
(see full_run.py).

In particular, we trigger

    > BranchCB features: at every node in the partial tree, i.e., until optimization stops
    > NodeCB features: at few points in time, {25, 50, 75, 100}% of rho time stamp.
    
    (e.g., when rho=20, we collect NodeCB features at {5, 10, 15, 20}% of time_limit, 
    where 20 = 100% of rho means just before stopping data collection)
    Note: there is no collection after the root node (the time-stamp is not mapped in TIME_NODES_DICT)
    
    In this way we can have more info about problems that are solved *before* tau, 
    for which the observed tree is the whole tree.
    
The information is saved directly into a (global) list, for the entire (partial) run.
At the end of the run, the list is converted to a single array and saved it in DATA_PATH/NPY/.

NOTE: for instances that are fully solved before tau (`trivial' ones), the last node will *not* be mapped by callbacks.
      This happens naturally, because solve() ends, and there is no way that node is going to enter 
      the BranchCB or the NodeCB. 
      In particular, the last datum about the gap will not be 0.0, and we will not have NodeCB as last row of the data.
      This is why final_info are collected and stored (for analysis and checks), though the information about the
      end of the resolution process is *not explicit* in the collected data.

Now the order of the (31) basic extracted features is important, and fixed as follows:

    GENERAL
0:  'global_cb_count'
1:  'times_called'
2:  'get_time' 
3:  'get_dettime'
4:  'elapsed'
5:  'det_elapsed'    

    BRANCH_CB ONLY
6:  'index'
7:  'parent_id'
8:  'parent_index'
9:  'node'
10: 'nodes_left'
11: 'objective'
12: 'iinf'                  **expensive**
13: 'best_integer'
14: 'best_bound'
15: 'itCnt'
16: 'gap'
17: 'num_nodes'
18: 'depth'
19: 'num_fixed_vars'        **expensive**
20: 'is_integer_feasible'
21: 'has_incumbent'          

    NODE_CB ONLY
22: 'open_nodes_len'
23: 'open_nodes_avg'
24: 'open_nodes_min'
25: 'open_nodes_max'
26: 'num_nodes_at_min'  
27: 'num_nodes_at_max'  
28: 'num_cuts'

    GENERAL
29: 'cb_time'
30: 'cb_dettime'

"""

"""
Parser definition.
"""

parser = argparse.ArgumentParser(description='Arg parser for data_run script.')

# Run identifiers: instance name, random seed, dataset
parser.add_argument(
    '--instance',
    type=str,
    required=True,
    help='Name of MIP instance to be processed (w/ full extension).'
)
parser.add_argument(
    '--seed',
    type=int,
    default=201610271,
    help='Random seed to be set in the solver. Default is 201610271 (CPLEX version 12.7.1).'
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
    help='Sets the time limit (secs). Default is 7200.'
)
parser.add_argument(
    '--node_limit',
    type=int,
    required=True,
    help='Node limit to be used in data collection. Obtained from full_run stats.'
)
parser.add_argument(
    '--rho',
    type=int,
    required=True,
    help='Percentage of time limits to be considered.'
)
parser.add_argument(
    '--label_time',
    type=float,
    required=True,
    help='Time(limit) w.r.t. which label and node_limit were computed.'
)

# Additional (only to store data)
parser.add_argument(
    '--label',
    type=int,
    required=True,
    help='Label in {0, 1} assigned during full_run w.r.t. label_time.'
)
parser.add_argument(
    '--trivial',
    type=bool,
    required=True,
    help='Trivial flag assigned during full_run.'
)
parser.add_argument(
    '--tau',
    type=float,
    required=True,
    help='Time corresponding to rho % of label_time.'
)
parser.add_argument(
    '--sol_time',
    type=float,
    required=True,
    help='Solution time of (instance, seed) during full_run.'
)

parser.add_argument(
    '--data_path',
    type=str,
    default=shared.DATA_PATH,
    help='Absolute path to data archive. Can be set within shared.py.'
)
parser.add_argument(
    '--inst_path',
    type=str,
    default=shared.INST_PATH,
    help='Absolute path to instances. Can be set within shared.py.'
)

ARGS = parser.parse_args()


"""
Custom classes definition.
"""


class GlobalCBCount:
    """
    A class setting a global count on Callbacks calls.
    """
    __counter = 0  # global counter of callbacks calls

    def __init__(self):
        GlobalCBCount.__counter += 1
        self.__cb_count = GlobalCBCount.__counter

    def get_cb_count(self):
        return self.__cb_count


class UserNodeIndex:
    """
    A class setting a global counter as unique index to each created node.
    """
    __index = -1  # so that root has index 0

    def __init__(self):
        UserNodeIndex.__index += 1
        self.__node_index = UserNodeIndex.__index

    def get_node_index(self):
        return self.__node_index


class UserNodeData:
    """
    A class defining a custom data-handle for nodes in the B&B tree.
    """
    def __init__(self, user_index, user_depth, user_parent_id, user_parent_index):
        self.user_index = user_index  # UserNodeIndex counter
        self.user_depth = user_depth  # depth will be updated recursively from parent to children
        self.user_parent_id = user_parent_id  # id of parent node
        self.user_parent_index = user_parent_index  # index of parent node


"""
Global definitions.
"""

# global definition of list, which will contain lists of length 31, i.e., the rows of the final array.
GLOBAL_LIST = list()

# the correct dictionary RHO_TAU_NODES is name_seed.pickle in ARGS.data_path/TIME_NODES_DICT/dataset_seed/
dict_dir = ARGS.data_path + "/TIME_NODES_DICT/" + ARGS.dataset + "_" + str(ARGS.seed) + "/"
dict_name = ARGS.instance.split('.')[0] + "_" + str(ARGS.seed) + ".pkl"

with open(os.path.join(dict_dir, dict_name), "rb") as p:
    RHO_TAU_NODES_DICT = pickle.load(p)
    nodeCB_FLAGS = OrderedDict()
    for eta in RHO_TAU_NODES_DICT[ARGS.rho][ARGS.tau]:
        if eta:  # map only node marks that are not None
            nodeCB_FLAGS[eta] = False
p.close()

# additional last NodeCB call at the end of the data collection
# (not met if node_limit corresponds to final number of nodes)
nodeCB_FLAGS[ARGS.node_limit] = False

# global instance of UserNodeData for the root node
root = UserNodeIndex()
print("\tRoot index is: {}".format(root.get_node_index()))
root_user_data = UserNodeData(user_index=root.get_node_index(), user_depth=0, user_parent_id=-1,
                              user_parent_index=None)

"""
Callbacks definition.
"""

# NOTE: there is no 'real' time_limit for data_run (enforce at 2h, with node_limit being effectively used).
# So it makes no sense to measure time_to_end.
# Elapsed time is meaningful, but accounts for data collection overhead as well.
# A more deterministic metric should be given by the number of nodes.


class MyDataNode(NodeCallback):
    """
    Custom subclass of NodeCallback.
    This callback will be used *before CPLEX enters a node*.
    """
    def __init__(self, env):
        NodeCallback.__init__(self, env)
        self.times_called = 0
        self.count_data = 0

    def __call__(self):
        self.times_called += 1
        gcb = GlobalCBCount()
        global nodeCB_FLAGS
        global GLOBAL_LIST

        t_stamp = self.get_time()
        det_stamp = self.get_dettime()

        s_elapsed = self.get_time() - self.get_start_time()
        det_elapsed = self.get_dettime() - self.get_start_dettime()

        # measure overhead of the whole CB call
        cb_start = self.get_time()
        cb_det_start = self.get_dettime()

        # at appropriate node marks, extract data
        for n in nodeCB_FLAGS.keys():
            if (self.get_num_nodes() >= n) and not nodeCB_FLAGS[n]:
                # NOTE: last flag will not be hit if it corresponds to the final number of nodes to solve a problem
                # (in case of problems that are solved before tau)
                print("*** NodeCB data call at num_nodes {}".format(self.get_num_nodes()))
                nodeCB_FLAGS[n] = True
                self.count_data += 1

                # print(self.get_num_nodes(), self.times_called, "NodeCB_FLAG", nodeCB_FLAGS[t])

                node_cb_list = list()
                # times and counters
                node_cb_list.append(gcb.get_cb_count())
                node_cb_list.append(self.times_called)
                node_cb_list.extend([t_stamp, det_stamp, s_elapsed, det_elapsed])

                # None for BranchCB entries
                node_cb_list.extend([None] * 16)

                # open nodes list
                open_list_obj = [self.get_objective_value(node) for node in range(self.get_num_remaining_nodes())]
                open_list_obj = np.asarray(open_list_obj)
                node_cb_list.extend([
                    len(open_list_obj),
                    np.mean(open_list_obj),
                    np.min(open_list_obj),
                    np.max(open_list_obj),
                    len(np.argwhere(open_list_obj == np.min(open_list_obj))),
                    len(np.argwhere(open_list_obj == np.max(open_list_obj)))
                ])
                # cuts
                cut_codes = [108, 111, 110, 107, 115, 120, 119, 117, 112, 122, 126, 133, 134, 135, 136]
                node_cb_list.append(np.sum([self.get_num_cuts(k) for k in cut_codes]))

                # cb_time, overhead of the callback call
                node_cb_list.extend([self.get_time() - cb_start, self.get_dettime() - cb_det_start])

                GLOBAL_LIST.append(node_cb_list)

                break


class MyDataBranch(BranchCallback):
    """
    Custom subclass of BranchCallback.
    This callback will be used *prior to branching* at a node in the branch and cut tree.
    """
    def __init__(self, env):
        BranchCallback.__init__(self, env)
        self.times_called = 0
        self.count_data = 0

    def __call__(self):
        self.times_called += 1
        gcb = GlobalCBCount()
        global ARGS
        global GLOBAL_LIST

        t_stamp = self.get_time()
        det_stamp = self.get_dettime()

        s_elapsed = self.get_time() - self.get_start_time()
        det_elapsed = self.get_dettime() - self.get_start_dettime()

        # measure overhead of the whole CB call
        cb_start = self.get_time()
        cb_det_start = self.get_dettime()

        # print("{}: Node - {}".format(self.times_called, self.get_node_ID()))

        # set user handle on new nodes, if cplex would create them
        # root node case is special
        data = self.get_node_data()  # UserNodeData object, or None for the root node

        if data is None:
            data = root_user_data  # now UserNodeData object as initialized above (global)

        if self.get_num_branches() > 0:
            for k in range(self.get_num_branches()):
                idx = UserNodeIndex().get_node_index()
                self.make_cplex_branch(
                    which_branch=k,
                    node_data=UserNodeData(
                        user_index=idx,
                        user_depth=data.user_depth+1,
                        user_parent_id=self.get_node_ID(),
                        user_parent_index=data.user_index
                    )
                )

        # features extraction
        if self.get_num_nodes() <= ARGS.node_limit:  # should not be necessary with node_limit enforced
            # NOTE: last node of node_limit is not recorded if it corresponds to the last node of the resolution,
            # because it will not enter BranchCB (in case of problems that are solved before tau)
            self.count_data += 1
            branch_cb_list = list()

            # times, indices, counters
            branch_cb_list.append(gcb.get_cb_count())
            branch_cb_list.append(self.times_called)
            branch_cb_list.extend([
                t_stamp,
                det_stamp,
                s_elapsed,
                det_elapsed
            ])
            branch_cb_list.extend([
                data.user_index,
                data.user_parent_id,
                data.user_parent_index
            ])

            # log info
            branch_cb_list.extend([
                self.get_node_ID(),
                self.get_num_remaining_nodes(),
                self.get_objective_value(),
                self.get_feasibilities().count(1),  # np.asarray ?
                self.get_incumbent_objective_value(),
                self.get_best_objective_value(),
                self.get_num_iterations(),
                self.get_MIP_relative_gap()
            ])
            # additional
            branch_cb_list.extend([
                self.get_num_nodes(),
                data.user_depth,
                ((np.asarray(self.get_upper_bounds()) - np.asarray(self.get_lower_bounds())) == 0).sum(),
                self.is_integer_feasible(),
                self.has_incumbent()
            ])

            # None for NodeCB entries
            branch_cb_list.extend([None] * 7)

            # cb_time, overhead of the callback call
            branch_cb_list.extend([
                self.get_time() - cb_start,
                self.get_dettime() - cb_det_start
            ])

            GLOBAL_LIST.append(branch_cb_list)


if __name__ == "__main__":

    import sys
    import time
    import os

    name = ARGS.instance.split('.')[0]
    inst_name_info = name + '_' + str(ARGS.seed) + '_' + str(ARGS.label_time) + '_' + str(ARGS.tau) + '_' + str(ARGS.rho)

    # sys.stdout = None

    os.chdir(ARGS.inst_path)
    pb = cplex.Cplex(ARGS.instance)

    print("\nPb name and type: {} {}".format(pb.get_problem_name(), pb.problem_type[pb.get_problem_type()]))
    print("# Variables: {}".format(pb.variables.get_num()))
    print("# Constraints: {}".format(pb.linear_constraints.get_num()))

    print("\nNode limit = {}".format(ARGS.node_limit))
    print("(rho, tl) = ({}, {}), tau = {}".format(ARGS.rho, ARGS.label_time, ARGS.tau))
    print("Solution time = {}".format(ARGS.sol_time))  # solution time of the full run
    print("Label = {}".format(ARGS.label))
    print("Trivial = {}".format(ARGS.trivial))

    # set solver parameters
    utilities.set_solver_data_run_parameters(pb, ARGS)

    # register callback
    branch_instance = pb.register_callback(MyDataBranch)
    node_instance = pb.register_callback(MyDataNode)

    try:
        t0, det_t0 = pb.get_time(), pb.get_dettime()
        print("\nInitial time-stamps: {} {}\n".format(t0, det_t0))
        pb.solve()
        elapsed = pb.get_time() - t0
        elapsed_ticks = pb.get_dettime() - det_t0
        status = pb.solution.get_status()
        num_nodes = int(pb.solution.progress.get_num_nodes_processed())
        itCnt = int(pb.solution.progress.get_num_iterations())
        gap = pb.solution.MIP.get_mip_relative_gap()
        print("\nStatus: {}".format(pb.solution.get_status()))
    except CplexSolverError as exc:  # data will be all zeros if exception is raised
        print("Exception raised during solve: {}".format(exc.args[2]))
        elapsed = None
        elapsed_ticks = None
        status = exc.args[2]  # error code
        num_nodes = None
        itCnt = None
        gap = None

    final_info = [elapsed, elapsed_ticks, status, num_nodes, itCnt, gap]

    print("\nMyDataBranch was called {} times".format(branch_instance.times_called))
    print("MyDataBranch count data: {}".format(branch_instance.count_data))
    print("\nMyDataNode was called {} times".format(node_instance.times_called))
    print("MyDataNode count data: {}".format(node_instance.count_data))

    print("\nNodeCB flags: ")
    print(nodeCB_FLAGS)

    print("\nFinal info of data_run: {}".format(final_info))

    # conversion to array
    global_arr = np.asarray(GLOBAL_LIST)
    print("Array shape is: ", global_arr.shape)

    # use np.savez to save into inst_name_info:
    npy_rho_dir = ARGS.data_path + '/NPY_RHO_' + str(int(ARGS.rho)) + '/' + ARGS.dataset + '_' + str(ARGS.seed) + '/'
    try:
        os.mkdir(npy_rho_dir)
    except OSError:
        if not os.path.isdir(npy_rho_dir):
            raise

    os.chdir(npy_rho_dir)
    np.savez_compressed(
        inst_name_info,
        ft_matrix=global_arr,
        name=name,
        seed=ARGS.seed,
        dataset=ARGS.dataset,
        rho=ARGS.rho,
        tau=ARGS.tau,
        node_limit=ARGS.node_limit,
        label_time=ARGS.label_time,
        label=ARGS.label,
        trivial=ARGS.trivial,
        sol_time=ARGS.sol_time,
        data_final_info=final_info,
    )

    pb.end()
