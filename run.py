# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

Script to run the CRAB model and save macro (model level)
and micro (agent level) outputs.

"""

import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from model import CRAB_Model


# -- READ FILES -- #
HH_attributes = pd.read_csv("Input/HH_attributes.csv", index_col=0)
firm_flood_depths = pd.read_csv("Input/Firm_attributes.csv", index_col=0)
PMT_weights = pd.read_csv("Input/PMT_weights.csv", index_col=0)

# -- MODEL PARAMETERS -- #
STEPS = 200
N_RUNS = 1
RANDOM_SEEDS = np.arange(0, 100, int(100/N_RUNS))


for n, seed in enumerate(RANDOM_SEEDS):
	print("RUN NR.", n+1)

	# -- INITIALIZE MODEL -- #
	model = CRAB_Model(seed, HH_attributes, firm_flood_depths, PMT_weights,
					   firms_RD=True, migration={"Regional": False, "RoW": True},
					   flood_when={},
					   CCA=False,
					   social_net=False)

	# -- RUN MODEL -- #
	for _ in tqdm(range(STEPS)):
		model.step()

	# -- COLLECT OUTPUT -- #
	if not os.path.isdir("results"):
		os.makedirs("results")
	model_vars = model.datacollector.get_model_vars_dataframe()
	model_vars.to_csv(f"results/model_vars_{seed}.csv", index=False)
	print(model_vars)
	agent_vars = model.datacollector.get_agent_vars_dataframe()
	agent_vars.to_csv(f"results/agent_vars_{seed}.csv")
	print(agent_vars)
