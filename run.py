# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

Script to run the CRAB model and save macro (model level)
and micro (agent level) outputs.

"""
import os
import time
import numpy as np

from tqdm.auto import tqdm

from model import CRAB_Model

STEPS = 300
N_RUNS = 10
RANDOM_SEEDS = np.arange(0, 100, int(100/N_RUNS))

for n, seed in enumerate(RANDOM_SEEDS):
	print("RUN NR.", n+1)

	tic = time.time()

	# -- INITIALIZE MODEL -- #
	model = CRAB_Model(seed)
	# -- RUN MODEL -- #
	for _ in tqdm(range(STEPS)):
		model.step()

	toc = time.time()
	print("TIME TO RUN MODEL: ", toc-tic)

	# -- COLLECT OUTPUT -- #
	if not os.path.isdir("results"):
		os.makedirs("results")
	model_vars = model.datacollector.get_model_vars_dataframe()
	model_vars.to_csv(f"results/model_vars_{seed}.csv", index=False)
	print(model_vars)
	agent_vars = model.datacollector.get_agent_vars_dataframe()
	agent_vars.to_csv(f"results/agent_vars_{seed}.csv")
	print(agent_vars)
