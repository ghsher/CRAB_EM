# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This file stores the model and agent variables for the datacollection
of the CRAB model.

"""

from CRAB_agents import *

REGION = 0

FIRM_TYPES = [
	Agriculture,
	Industry,
	Construction,
	Transport,
	Utilities,
	Private_services,
	Public_services,
	Wholesale_Retail,
    C26,
    # C27,
    # C28,
    # C29,
    # C30,
]

model_vars = {}
agent_vars = {}

# TODO :: replace this.
# Jan's right the DC is truly shitty (esp. the lack of ability to group and the incredible waste of space)

agent_vars['Type'] = lambda a: a.__class__.__name__
# agent_vars['Region'] = lambda a: getattr(a, "region", None)

######################################################
### 0. POPULATION (outcome & endogenous predictor) ###
######################################################

# Agents report Type, so this can be post-processed

#########################################
### 1. ECONOMIC PERFORMANCE (outcome) ###
#########################################

#	a. Production (per type of firm)
for type in FIRM_TYPES:
	name = type.__name__
	model_vars[f'{name} Production Made'] = lambda m: sum([
			getattr(f, "production_made", 0) for f in 
			m.get_firms_by_type(type, REGION)
		])

#	b. Employment status (per household)
agent_vars['Employed'] = \
	lambda a: True if getattr(a, "employer", None) is not None else False

##################################################
### 2. WEALTH (outcome & endogenous predictor) ###
##################################################

#	a. Net worth / Accrued wealth 
# 	 TODO: This is missing the value of one's adaptation measures, in theory
agent_vars['Net Worth'] = lambda a: getattr(a, "net_worth", None)

#	b. House value 
# 	 TODO: This is missing the value of one's adaptation measures, in theory
agent_vars['House Value'] = lambda a: getattr(a, "house_value", None)

#	c. Income (we'll later extract this just for households)
agent_vars['Wage'] = lambda a: getattr(a, "wage", None)

##################################################
### 3. FIRM CRITICALITY (endogenous predictor) ###
##################################################

# 	a. Firm size
#		Will be used to proxy industry competition/agglomeration
agent_vars['Firm Size'] = lambda a: getattr(a, 'size', None)

# model_vars['Population: Small Firms'] = lambda m: len([
# 			f for f in m.get_firms(REGION)
# 			if f.size > SIZE_THRESHOLDS['Small']
# 			and f.size <= SIZE_THRESHOLDS['Medium']
# 		])
# model_vars['Population: Medium Firms'] = lambda m: len([
# 			f for f in m.get_firms(REGION)
# 			if f.size > SIZE_THRESHOLDS['Medium']
# 			and f.size <= SIZE_THRESHOLDS['Large']
# 		])
# model_vars['Population: Large Firms'] = lambda m: len([
# 			f for f in m.get_firms(REGION)
# 			if f.size > SIZE_THRESHOLDS['Large']
# 		])

###############################
### 4. INEQUALITY (outcome) ###
###############################

# Agents report Wage and Net worth, so Gini can be post-processed
# Distributional effects can also be measured by keeping agent data
#  partially disaggregated by adaptive capacity & income quartile


#################################
### 5. FLOOD IMPACT (outcome) ###
#################################

#	a. Flood damages
#	    Since we want to be able to normalize Damages by income or other 
#		measures, we should track this at an agent level
#		TODO: Check with Liz about implementation (doesn't update until all repairs are done)
#				(see issue_archive.md)
agent_vars['Damages'] = lambda a: getattr(a, 'monetary_damage', None)

#	b. Net worth recovery (by household)
# TODO: For now, handle in post

#	c. Wage level recovery (by household)
# TODO: For now, handle in post

# Agents report Wage and Net worth, so Gini can be post-processed
# Distributional effects can also be measured by keeping agent data
#  partially disaggregated by adaptive capacity & income quartile

#########################
### 6. DEBT (outcome) ###
#########################

#	a. Firm debt
agent_vars['Debt'] = lambda a: getattr(a, 'debt', None)

#	b. Government deficit
# model_vars['Deficit'] = lambda m: m.get_deficit() # TODO

#################################################
### 7. OTHERS (usually endogenous predictors) ###
#################################################

# 	a. Minimum wage
model_vars['Minimum Wage'] = lambda m: m.governments[REGION].min_wage
