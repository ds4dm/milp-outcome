## Learning MILP resolution outcomes before reaching time-limit 

Given a MILP instance and a timelimit (*TL*), a general question about its resolution process asks at some time *0<t<TL*
whether the instance will be solved to (proven) optimality within that *TL* or not. 
The prediction we aim at is one that takes as input a single data-point summarizing the 
**evolution of the partial B&B search tree** up to time *t*, and outputs a yes/no response.
The framework is that of **binary classification**, with features that should embed time-series (temporal) data, 
collected at each node of the partial tree.

This work can be found [here](https://link.springer.com/chapter/10.1007/978-3-030-19212-9_18), and can be cited as 

```
@InProceedings{10.1007/978-3-030-19212-9_18,
  author="Fischetti, Martina and Lodi, Andrea and Zarpellon, Giulia",
  editor="Rousseau, Louis-Martin and Stergiou, Kostas",
  title="Learning MILP Resolution Outcomes Before Reaching Time-Limit",
  booktitle="Integration of Constraint Programming, Artificial Intelligence, and Operations Research",
  year="2019",
  publisher="Springer International Publishing",
  address="Cham",
  pages="275--291",
  isbn="978-3-030-19212-9"
}
```

### Notes

We use Python 3.5 and CPLEX 12.7.1.0 as MILP solver. The `numpy` and `pandas` libraries are employed for data
manipulation, with `scikit-learn` for machine learning experiments.

The code automatizes the collection of several samples from each MILP instance, 
considering multiple values of *TL* at a time, and streamlining as much as possible tedious steps for batches of experiments. 
The majority of the code is thus composed of Python scripting to obtain data and organize inputs/outputs, 
and experiments are only a tiny fraction of the process.

To avoid biasing the data with the overhead of computing features, we divide the implementation into two main steps:
 - _Label computation_: a MILP instance is run to compute labels and record the number of nodes explored up to time *t*;
 - _Data collection_: the instance is run again with, this time with the previously obtained node-limit. 
 Data extraction happens in this run: 31 raw indicators are collected using Branch and Node callbacks of CPLEX, 
 producing a time-series for the partial run. This time-series is what later on is summarized into proper features for learning.
 

#### Experiments pipeline

1. `full_run.py`<br/>The _label computation_ step: runs an (_instance_, _seed_) pair with a single (the longest specified) time limit (usually 2h).
Only basic info are collected during the run at predefined time stamps that depend on *TL* and *t*. 
In particular, the number of nodes explored at time *t* is recorded.
Within the same run, info are collected for other specified intermediate (i.e., smaller) values of *TL* as well.

2. `read_stats.py`<br/>Pure utility scripting: reads outputs of `full_run.py` to create data summaries for later analysis, 
and automatizes generation of command lines to launch data collection runs.

3. `data_run.py`<br/>The _data collection_ step: runs an (_instance_, _seed_) pair to extract 31 basic indicators 
from the B&B tree via Branch and Node callbacks, up to the specified node limit (obtained from `full_run.py`).
The resulting time-series are saved in `.npz` files. 

4. `features_select_37.py`<br/>Feature computation of the 37 features which were selected for the experiments reported in the paper, 
which are documented in `doc/features_select_37.md`.
Time-series data in each `.npz` file is processed and summarized into a single vector of features, 
and finally a dataset for learning is saved.
Note that feature design is an iterative process: we initially developed more features (`features_all.py`) and subsequently 
added some more when experimenting with machine learning models (`learning.ipynb`). 

Data inspection, basic cleaning and a simple loop for classification experiments with `scikit-learn` are in `learning.ipynb`.

Finally, `utilities.py` contains utility functions to, e.g., set the solver parameters and setup outputs directories.

#### Questions?
Please feel free to submit a GitHub issue if you have any questions or find any bugs. 
We do not guarantee any support or maintenance, but we will do our best if we can help.
