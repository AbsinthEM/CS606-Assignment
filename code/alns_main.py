import os
import argparse

import numpy.random as rnd
from operators import destroy_random, destroy_time_blocks, destroy_cost, repair_greedy, repair_time_blocks, repair_regret
from psp import PSP, Parser
from src.alns import ALNS
from src.alns.criteria import *
from src.helper import save_output
from src.settings import DATA_PATH


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    json_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(json_file)
    psp = PSP(parsed.name, parsed.workers, parsed.tasks, parsed.Alpha)

    # construct random initialized solution
    psp.random_initialize(seed)

    print("Initial solution objective is {}.".format(psp.objective()))

    # Generate output file
    save_output("Han_Zhuoyan_ALNS", psp, "initial")  # // Modify with your name

    # ALNS
    random_state = rnd.RandomState(seed)

    alns = ALNS(random_state)

    # -----------------------------------------------------------------
    # // Implement Code Here
    # You should add all your destroy and repair operators here
    # add destroy operators
    alns.add_destroy_operator(destroy_random)
    alns.add_destroy_operator(destroy_time_blocks)
    alns.add_destroy_operator(destroy_cost)

    # // add repair operators
    alns.add_repair_operator(repair_greedy)
    alns.add_repair_operator(repair_time_blocks)
    alns.add_repair_operator(repair_regret)
    # -----------------------------------------------------------------

    # run ALNS & Select Criterion
    criterion = HillClimbing()

    # Weights for global best, accepted, simulation accepted, rejected
    omegas = [10, 5, 3, 1]  # // Select the weights adjustment strategy
    # Decay parameter for weight adjustment
    lambda_ = 0.8  # // Select the decay parameter
    result = alns.iterate(
        psp, omegas, lambda_, criterion, iterations=10000, collect_stats=True
    )  # // Modify number of ALNS iterations as you see fit

    # result
    solution = result.best_state
    objective = solution.objective()
    print("Best heuristic objective is {}.".format(objective))

    # visualize final solution and generate output file
    save_output("Han_Zhuoyan_ALNS", solution, "solution")  # // Modify with your name
