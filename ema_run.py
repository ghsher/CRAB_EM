# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: ghsher

Script to run any number of experiments on the CRAB model
using the EMA Workbench. 

Information about the EMA Workbench is available at
 https://emaworkbench.readthedocs.io/en/latest/

"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from ema_workbench import (
    ema_logging,
    SequentialEvaluator,
    MultiprocessingEvaluator,
    MPIEvaluator,
    Scenario,
    save_results,
)

from CRAB_agents import *
from ema_data_collection import *
from ema_model import get_EMA_CRAB_model

FLOOD_INTENSITIES = [
    [3000]
]

if __name__ == "__main__":
    ##################
    ### USER INPUT ###
    ##################

    # Set up arg parser
    parser = argparse.ArgumentParser(
                    prog='CRAB Model EMA Workbench Runner',
                    description='Performs experiments using the CRAB model')
    parser.add_argument('-S', '--starting_seed',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('-N', '--num_reps',
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument('-T', '--time_horizon',
                        type=int,
                        default=120, # 5 year burn-in + 25 year run-time
                        required=False)
    args = parser.parse_args()

    # Parse arguments
    N_REPLICATIONS = args.num_reps
    STARTING_SEED = args.starting_seed
    SEEDS = np.arange(STARTING_SEED, STARTING_SEED+N_REPLICATIONS)
    STEPS = args.time_horizon

    # Clean up, for memory reasons
    del parser, args

    ######################
    ### SET UP EMA RUN ###
    ######################

    # Runtime output settings
    # ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
    ema_logging.LOG_FORMAT = "[EMA] %(message)s"
    ema_logging.log_to_stderr(ema_logging.INFO, pass_root_logger_level=True) # Uncomment for MPI
    
    # Fetch an EMA_workbench Model object
    model = get_EMA_CRAB_model(
        flood_intensities=FLOOD_INTENSITIES,
        seeds=SEEDS,
        steps=STEPS
    )

    # Read Scenarios from CSV
    scenarios = pd.read_csv('data/2_scenarios_0611.csv', # XXX: Edit here
                             index_col=False) 
    scen_list = []
    for idx, scen in scenarios.iterrows():
        scen_list.append(Scenario(idx, **scen))

    del scenarios

    N_SCENARIOS = len(scen_list)

    #################
    ### RUN MODEL ###
    #################

    # with MPIEvaluator(model) as evaluator:
    with SequentialEvaluator(model) as evaluator: # left in for local testing
        results = evaluator.perform_experiments(
            scenarios=scen_list,
        )

    #######################
    ### PROCESS OUTPUTS ###
    #######################

    # Process replications: for now, save each separately
    #  Have to extend the experiments df to match sure index still matches
    #  with the "tall" outcomes arrays
    experiments, outcomes = results
    new_exps = {col:[] for col in experiments}
    new_exps['seed'] = []
    new_outs = {out:[] for out in outcomes}

    # Create new dicts as baseline for "tall" dataset
    for rep in range(N_REPLICATIONS):
        for scen in range(N_SCENARIOS):
            for col in experiments:
                new_exps[col].append(experiments.loc[scen, col])
            new_exps['seed'].append(STARTING_SEED+rep)
            for out in outcomes:
                new_outs[out].append(list(outcomes[out][scen][rep]))
    # Make sure outcomes are saved as np arrays
    for out in new_outs:
        new_outs[out] = np.array(new_outs[out])

    # Cast experiments to Pandas DF
    experiments = pd.DataFrame(new_exps)
    outcomes = new_outs
    del new_exps # delete to save memory for disk write
    del new_outs #  (not sure this is necessary)

    # Save results !!
    results = experiments, outcomes

    DATE = datetime.now().strftime("%m%d")
    filename = f"results/{N_SCENARIOS}_scen__{N_REPLICATIONS}_reps__{DATE}.tar.gz"
    save_results(results, filename)
