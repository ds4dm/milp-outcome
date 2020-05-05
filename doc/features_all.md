### Features list

#### `from_last_seen_data` (@last BranchCB and NodeCB call)
1. `nodeID / num_nodes`: Sequence # of current node compared to all processed ones
2. `nodeID / nodes_left`: Sequence # of current node compared to # of nodes left (open)
3. `nodes_left`: Number of nodes left 
4. `best_integer`: Value of best integer (incumbent)
5. `best_bound`: Value of best bound
6. `itCnt`: Iteration count
7. `gap`: Gap 
8. `has_incumbent`: Has incumbent? (boolean, referred to entire tree)
9. `num_nodes`: Number of nodes processed so far
10. `objective / best_integer`: Ratio objective over incumbent
11. `best_bound / objective`: Ratio best bound over objective
12. `best_bound / best_integer`: Ratio best bound over incumbent
13. `open_nodes_len`: Length of open nodes list (might slightly different from `num_nodes`)
14. `open_nodes_max`: Max objective value in open nodes list
15. `open_nodes_min`: Min objective value in open nodes list
16. `open_nodes_avg`: Average objective value in open nodes list
17. `num_nodes_at_max / open_nodes_len`: Ratio of nodes attaining max objective in open nodes list, over total number
18. `num_nodes_at_min / open_nodes_len`: Ratio of nodes attaining min objective in open nodes list, over total number
19. `open_nodes_max / best_integer`: Ratio max objective of open nodes list over incumbent
20. `open_nodes_min / best_integer`: Ratio min objective of open nodes list over incumbent
21. `open_nodes_min / open_nodes_max`: Ratio max over min objective in open nodes list
22. `objective / open_nodes_max`: Ratio objective over max objective in open nodes list
23. `open_nodes_min / objective`: Ratio min objective of open nodes list over objective
24. `num_cuts`: Number of cuts applied

#### `about_pruned`
1. `pruned`: Total number of pruned nodes
2. `pruned / num_nodes`: Pruned throughput, over number of processed nodes
3. `pruned / nodes_left`: Pruned throughput, over number of nodes left

#### `about_nodes_left`
1. `nodes_left / num_nodes`: Nodes left throughput, over number of processed nodes
2. `nodes_left / max nodes_left`: Ratio nodes left over maximum seen number of nodes left
3. `4x nodes_left slopes`: Rate of change (slopes) of `nodes_left` in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
4. ...in 5-10 %
5. ...in 10-15 % 
6. ...in 15-20 %

#### `about_iinf`
1. `max, min, avg iinf / total discrete`: Ratio of max, min, avg `iinf` over total number of discrete variables
2. ...min
3. ...avg
4. `iinf / total discrete`: Ratio of last seen `iinf` over total number of discrete variables
5. `# nodes with iinf in 0.05 quantile / total`: Ratio of # nodes with `iinf` in 0.05 quantile over total number of calls
6. `last iinf / 0.05 quantile value`: Ratio of last `iinf` over value of 0.05 quantile
7. `distance to last seen in quantile to end / total`: Ratio of distance between last seen node in 0.05 quantile to end of data collection over total number of calls (rows)

#### `about_itcnt`
1. `ItCnt / num_nodes`: ItCnt throughput, over number of processed nodes

#### `about_integer_feasible`
1. `# integer feasible found / total`: Ratio between # of integer feasible found over total # of processed nodes
2. `incumbent before integer feasible node`: Was an incumbent found before an integer feasible node? (boolean)
3. `depth first incumbent`: Depth of first incumbent

#### `about_incumbent`
1. `# of incumbent updates`: Total number of incumbent updates
2. `num_updates / num_nodes`: Incumbent updates throughput, over number of processed nodes
3. `max_improvement, min_improvement, avg_improvement`: Max, min and average incumbent improvement (abs value)
4. ...min
5. ...avg
6. `avg incumbent improvement / first incumbent value`: Avg incumbent improvement over value of first incumbent
7. `max, min, avg distance between past incumbent updates`: Max, min, average distance between two consecutive incumbent updates
8. ...min
9. ...avg
10. `distance from last update`: Distance between the last incumbent update and last explored node

#### `about_best_bound`
1. `# of best_bound updates`: Total number of best bound updates
2. `num_updates / num_nodes`: Best bound updates throughput, over number of processed nodes
3. `max_improvement, min_improvement, avg_improvement`: Max, min and average best bound improvement (abs value)
4. ...min
5. ...avg
6. `avg best_bound improvement / first best_bound value`: Avg best bound improvement over value of first best bound
7. `max, min, avg distance between past best_bound updates`: Max, min, average distance between two consecutive best bound updates
8. ...min
9. ...avg
10. `distance from last update`: Distance between the last best bound update and last explored node

#### `about_objective`
1. `|current objective - root objective`: Difference between objective (of last BranchCB call) and root objective (abs value)
2. `root objective / current objective`: Ratio between root objective and last objective
3. `# nodes with objective in 0.05 quantile (0.95) / total`: Ratio of # nodes with `objective` in 0.05 quantile over total number of calls
4. `last objective / value of 0.05 quantile`: Ratio of last objective over value of 0.05 (0.95) quantile
5. `|last best_integer - value of 0.05 quantile|`: Difference between incumbent and value of 0.05 quantile (abs value)
6. `|last best_bound - value of 0.05 quantile|`: Difference between best bound and value of 0.05 quantile
7. `distance to last seen in quantile to end / total`: Ratio of distance between last seen node in 0.05 quantile to end of data collection over total number of calls (rows)

#### `about_active_constraints`
1. `num_active_constraints / total number of constraints`: Ratio of active constraints over total number of constraints
2. `num_active_constraints / num_incumbent_active_constraints`: Ratio of active constraints (at last BranchCB call) over incumbent active constraints

#### `about_fixed_vars`
1. `max, min / total # of variables`: Max and min # of fixed variables over total # of variables
2. ...min
3. `last num_fixed_vars / total # of variables`: Ratio of last # of fixed variables over total # of variables
4. `# nodes with num_fixed_vars in 0.05 quantile (0.95) / total`: Ratio of # nodes with `num_fixed_vars` in 0.05 quantile over total number of calls
5. `last num_fixed_vars / value of 0.05 quantile`: Ratio of last `num_fixed_vars` over value of 0.05 (0.95) quantile
6. `distance to last seen node in quantile to end / total`: Ratio of distance between last seen node in 0.05 quantile to end of data collection over total number of calls (rows)

#### `about_depth` (of the explored tree)
1. `d_t`: Max depth 
2. `l_t`: Last full level 
3. `max_width over levels`: Max width, across all levels
4. `# levels at max_width`: Number of levels at max width
5. `b_t / d_t`: Waist of the tree over max depth
6. `b_t / l_t`: Waist of the tree over last full level
7. `wl_t`: Waistline (width at the waist)
8. `n_t`: Total number of considered nodes (corresponds to # BranchCB calls)

#### `about_dives` (or more generally jumps)
1. `max, min, avg len of dives`: Max, min, average length of dives
2. ...min
3. ...avg
4. `# changes in depth with abs > 1`: # of changes in depth with abs > 1 (identify backtracks between branches and in particular end of dives)
5. `num_nodes at end of first dive / num_nodes at last seen node`: Ratio of length of first dive over total number of processed nodes
6. `depth of first dive / max depth in the partial tree`: Ratio of depths of first dive over max depth observed in tree
7. `num_nodes at end of last dive / num_nodes at last seen node`: Ratio of length of last dive/jump over # processed nodes
8. `depth of last dive / max depth in the partial tree`: Ratio of depths of last dive over max depth observed in tree
9. `max, min, avg distance between depth jumps with abs > 1`: Max, min, average distance between consecutive jumps with abs > 1
10. ...min
11. ...avg
12. `distance between last jump and last node explored`: Distance between last jump and last node explored

#### `about_integral`
1. `primal integral`: Primal integral (if best known solution is available)
2. `primal-dual integral`: Primal-dual integral (with current `best_bound` and `incumbent`)

#### `about_cuts`
1. `num_cuts applied after root node`: Number of cuts applied after root node
2. `total / root node cuts`: Total number of cuts over number of cuts at root
3. `max, min, avg deltas of cuts additions`: Max, min, average addition of cuts (between two NodeCB calls)
4. ...min
5. ...avg

#### `about_open_nodes`
1. `4x open_nodes_max slopes`: Rate of change (slopes) of `open_nodes_max` in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
2. ...in 5-10 %
3. ...in 10-15 %
4. ...in 15-20%
5. `4x open_nodes_min slopes`: Rate of change (slopes) of `open_nodes_min` in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
6. ...in 5-10 %
7. ...in 10-15 %
8. ...in 15-20 %
9. `4x num_nodes_at_max slopes`: Rate of change (slopes) of `num_nodes_at_max` in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
10. ...in 5-10 %
11. ...in 10-15 %
12. ...in 15-20 %
13. `4x num_nodes_at_min slopes`: Rate of change (slopes) of `num_nodes_at_min` in 0-5, 5-10, 10-15, 15-20 % (to end of partial run)
14. ...in 5-10 %
15. ...in 10-15 %
16. ...in 15-20 %

#### `about_time`
1. `elapsed`: Elapsed time
2. `remaining`: Remaining time