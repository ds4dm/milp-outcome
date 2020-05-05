import os

import shared

"""
Collection of auxiliary functions.
"""


def set_solver_full_run_parameters(c, info_str, dir_str, args):
    """
    :param c: cplex.Cplex() instantiation
    :param info_str: str, "name_seed" of run identifiers
    :param dir_str: str, "dataset_seed" of batch identifiers
    :param args: ArgumentParser.pars_args() instantiation
    Set the proper solver parameters for full_run.
    ----------
    NOTE: a single timelimit of 2h is enforced (the longest of the one we are interested in).
    NOTE2: no node-limit is specified for full_run.
    """

    try:
        os.mkdir(args.data_path + "/RES/" + dir_str)
    except OSError:
        if not os.path.isdir(args.data_path + "/RES/" + dir_str):
            raise

    try:
        os.mkdir(args.data_path + "/ERR/" + dir_str)
    except OSError:
        if not os.path.isdir(args.data_path + "/ERR/" + dir_str):
            raise

    c.set_results_stream(os.path.join(args.data_path + '/RES/' + dir_str, info_str + '.res'))
    c.set_log_stream(None)
    c.set_warning_stream(None)
    c.set_error_stream(os.path.join(args.data_path + '/ERR/' + dir_str, info_str + '.err'))

    c.parameters.randomseed.set(args.seed)

    # c.parameters.mip.interval.set(10)  # automatic, set interval for node log
    c.parameters.mip.display.set(2)

    # node, time and memory limits
    c.parameters.timelimit.set(args.time_limit)
    c.parameters.mip.limits.treememory.set(10000)  # 10GB (10000 MB)

    # presolve
    c.parameters.preprocessing.presolve.set(1)  # default, apply presolve

    # probing at default
    # polishing is not invoked by default
    # primal heuristics on
    # cuts on

    # need to use traditional and sequential branch-and-cut to allow for control callbacks
    c.parameters.mip.strategy.search.set(c.parameters.mip.strategy.search.values.traditional)
    c.parameters.threads.set(1)

    try:
        os.mkdir(args.data_path + "/PAR/" + dir_str)
    except OSError:
        if not os.path.isdir(args.data_path + "/PAR/" + dir_str):
            raise

    c.parameters.write_file(os.path.join(args.data_path + '/PAR/' + dir_str, info_str + '.par'))

    return


def set_solver_data_run_parameters(c, args):
    """
    :param c: cplex.Cplex() instantiation
    :param args: ArgumentParser.pars_args() instantiation
    Set the proper solver parameters for data_run, i.e., data collection.
    Basically the same setting of full_run, apart from output streams, parameters log and *node limit*.
    ----------
    NOTE: a single timelimit on 2h is enforced.
    NOTE2: node-limit is specified for data_run.

    """
    c.set_results_stream(None)
    c.set_log_stream(None)
    c.set_warning_stream(None)
    c.set_error_stream(None)

    c.parameters.randomseed.set(args.seed)

    # c.parameters.mip.interval.set(10)  # automatic, set interval for node log
    c.parameters.mip.display.set(2)

    # node, time and memory limits
    c.parameters.timelimit.set(args.time_limit)
    c.parameters.mip.limits.nodes.set(args.node_limit)  # sets the maximum number of nodes *solved* (processed)
    c.parameters.mip.limits.treememory.set(10000)  # 10GB (10000 MB)

    # presolve
    c.parameters.preprocessing.presolve.set(1)  # default, apply presolve

    # probing at default
    # polishing is not invoked by default
    # primal heuristics on
    # cuts on

    # need to use traditional and sequential branch-and-cut to allow for control callbacks
    c.parameters.mip.strategy.search.set(c.parameters.mip.strategy.search.values.traditional)
    c.parameters.threads.set(1)

    return


def write_data_jobs_args(data, file_name, dataset_name):
    """
    :param data: DataFrame object, read from /RUNS_INFO/ + rho_dataset_seed.txt file (eventually part of it)
    :param file_name: name of the file to be created (with path, normally under /DATA_ARGS/)
    :param dataset_name: str, name of dataset
    :return: .txt file containing command line arguments to run data_run.py
    The file will contain one line for each row in data.
    """
    # if from original txt file, header should be
    # header = ["NAME", "SEED", "RHO", "TL", "TAU", "NODES_TAU", "GAP_TAU", "B_CALLS_TAU", "LABEL", "TRIVIAL",
    #           "GAP", "SOL_TIME", "NODES", "B_CALLS", "STATUS"]

    count_rows = 0
    with open(file_name, 'a') as args_append:
        for index, row in data.iterrows():
            count_rows += 1
            data_args = "--instance " + str(row['NAME'])+".mps.gz" +\
                        " --seed " + str(row['SEED']) + " --dataset " + dataset_name + \
                        " --rho " + str(int(row['RHO'])) + " --tau " + str(row['TAU']) +\
                        " --node_limit " + str(row['NODES_TAU']) + \
                        " --label_time " + str(row['TL']) + " --sol_time " + str(row['SOL_TIME']) +\
                        " --label " + str(row['LABEL']) + " --trivial " + str(row['TRIVIAL']) +\
                        "\n"
            args_append.write(data_args)
    args_append.close()
    print("Total # of rows written: {}".format(count_rows))
    return


def dir_setup(parent_path_inst_dir, rhos):
    """
    :param parent_path_inst_dir: absolute path to DATA ARCHIVE
    :param rhos: list, of explored rho values
    Create directories for various outputs.
    """

    os.chdir(parent_path_inst_dir)

    # general shared directories: ERR, OUT, RES, STAT, PAR, LABEL_NL, NPY
    try:
        os.mkdir("ERR")
    except OSError:
        if not os.path.isdir("ERR"):
            raise
    try:
        os.mkdir("OUT")
    except OSError:
        if not os.path.isdir("OUT"):
            raise
    try:
        os.mkdir("RES")
    except OSError:
        if not os.path.isdir("RES"):
            raise
    try:
        os.mkdir("STAT")
    except OSError:
        if not os.path.isdir("STAT"):
            raise
    try:
        os.mkdir("PAR")
    except OSError:
        if not os.path.isdir("PAR"):
            raise
    try:
        os.mkdir("RUNS_INFO")
    except OSError:
        if not os.path.isdir("RUNS_INFO"):
            raise
    try:
        os.mkdir("DATA_ARGS")
    except OSError:
        if not os.path.isdir("DATA_ARGS"):
            raise

    try:
        os.mkdir("TIME_NODES_DICT")
    except OSError:
        if not os.path.isdir("TIME_NODES_DICT"):
            raise

    for rho in rhos:
        try:
            os.mkdir("NPY_RHO_"+str(int(rho)))
        except OSError:
            if not os.path.isdir("NPY_RHO_"+str(int(rho))):
                raise

    return


if __name__ == "__main__":

    import pandas as pd
    import glob

    # parameters
    dataset = "SingleTest"
    seed = 201610271
    info_file = "../data_dir/RUNS_INFO/20_SingleTest_201610271.txt"

    # write arguments for data_run jobs
    cols_to_keep = ["NAME", "SEED", "NODES_TAU", "TL", "RHO", "LABEL", "TRIVIAL", "TAU", "SOL_TIME", "STATUS"]

    # to iterate through multiple txt files, use:
    # os.chdir(shared.DATA_PATH + '/RUNS_INFO/')
    # for f in glob.glob('*.txt'):

    df = pd.read_csv(info_file, sep="\t", usecols=cols_to_keep)
    file_name = dataset + '_' + str(seed) + '_args_jobs.txt'
    args_file = os.path.join(shared.DATA_PATH + "/DATA_ARGS/", file_name)

    # remove from df error runs
    # errors are identified as STATUS values of 4 digits, and the max solution status is 133
    mask = (df['STATUS'] <= 133)
    df_sel = df.loc[mask]
    print(
        "\nDF was reduced from {} to {} entries ({} error instances were removed)".format(df.shape[0],
                                                                                          df_sel.shape[0],
                                                                                          df.shape[0] - df_sel.shape[
                                                                                              0])
    )

    write_data_jobs_args(df_sel, args_file, dataset)
