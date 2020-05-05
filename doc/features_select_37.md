### Features list (37 selected for learning)
(ipynb ordering)

#### `from_last_seen_data`
7. `gap`: Gap 
12. `best_bound / best_integer`: Ratio best bound over incumbent
17. `num_nodes_at_max / open_nodes_len`: Ratio of nodes attaining max objective in open nodes list, over total number
18. `num_nodes_at_min / open_nodes_len`: Ratio of nodes attaining min objective in open nodes list, over total number
19. `open_nodes_max / best_integer`: Ratio max objective of open nodes list over incumbent
20. `open_nodes_min / best_integer`: Ratio min objective of open nodes list over incumbent

#### `about_pruned`
2. `pruned / num_nodes`: Pruned throughput, over number of processed nodes
3. `pruned / nodes_left`: Pruned throughput, over number of nodes left

#### `about_nodes_left`
2. `nodes_left / max nodes_left`: Ratio nodes left over maximum seen number of nodes left

#### `about_iinf`
1. `max, min, avg iinf / total discrete`: Ratio of max, min, avg `iinf` over total number of discrete variables
2. ...min
3. ...avg
5. *** `# nodes with iinf in 0.05 quantile / total`: Ratio of # nodes with `iinf` in 0.05 quantile over total number of calls

#### `about_itcnt`
1. `ItCnt / num_nodes`: ItCnt throughput, over number of processed nodes

#### `about_integer_feasible`
2. `incumbent before integer feasible node`: Was an incumbent found before an integer feasible node? (boolean)

#### `about_incumbent`

2. `num_updates / num_nodes`: Incumbent updates throughput, over number of processed nodes
A5. `avg_improvement / last_seen_incumbent`: average incumbent improvement over incumbent value
A9. `avg_distance_updates / num_branched_nodes`: average distance between incumbent updates over total number of branched nodes
A10. `distance from last update / avg_distance_updates`: distance from last update over average update distance

#### `about_best_bound`
2. `num_updates / num_nodes`: Best bound updates throughput, over number of processed nodes
A5. `avg_improvement / last_seen_best_bound`: average best bound improvement over best bound value
A9. `avg_distance_updates / num_branched_nodes`: average distance between best bound updates over total number of branched nodes
A10. `distance from last update / avg_distance_updates`: distance from last update over average update distance

#### `about_objective`

3. *** `# nodes with objective in 0.05 quantile (0.95) / total`: Ratio of # nodes with `objective` in 0.05 quantile over total number of calls
5. `|last best_integer - value of 0.05 quantile|`: Difference between incumbent and value of 0.05 quantile (abs value)
6. `|last best_bound - value of 0.05 quantile|`: Difference between best bound and value of 0.05 quantile

#### `about_active_constraints`

#### `about_fixed_vars`
1. `max, min / total # of variables`: Max and min # of fixed variables over total # of variables
2. ...min
4. *** `# nodes with num_fixed_vars in 0.05 quantile (0.95) / total`: Ratio of # nodes with `num_fixed_vars` in 0.05 quantile over total number of calls
6. `distance to last seen node in quantile to end / total`: Ratio of distance between last seen node in 0.05 quantile to end of data collection over total number of calls (rows)

#### `about_depth` (of the explored tree)
A1. `max_depth / num_nodes`: max depth over total number of processed nodes
A2. `last_full_level / max_depth`: last full level over max depth

5. `b_t / d_t`: Waist of the tree over max depth

#### `about_dives` (or more generally backtracks)

1. `max len of dives`: max length of dives
3. `avg len of dives` : average length of dives

A4. `num_backtracks / total_branched`: # of changes in depth over total number of branched nodes

#### `about_integral`

2. `primal-dual integral`: Primal-dual integral (with current `best_bound` and `incumbent`)

#### `about_cuts`

#### `about_open_nodes`

#### `about_time`
