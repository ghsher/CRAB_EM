# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This file stores the model and agent variables for the datacollection
of the CRAB model.

"""

from CRAB_agents import *

REGION = 0

model_vars = {
			  "n_agents": 
			  		lambda m: m.schedule.get_agent_count(),
			  "n_households": 
			  		lambda m: len(m.get_households(REGION)),
			  "n_cap_firms": 
		  			lambda m: len(m.get_firms_by_type(CapitalFirm, REGION)),
			  "n_cons_firms": 
			  		lambda m: len(m.get_firms_by_type(ConsumptionGoodFirm, REGION)),
			  "n_serv_firms": 
			  		lambda m: len(m.get_firms_by_type(ServiceFirm, REGION)),
			  "HH consumption": 
			  		lambda m: sum(hh.consumption for hh in m.get_households(REGION)),
			  "Regional demand":
			  		lambda m: sum(m.governments[REGION].regional_demands.values()),
			  "Export demand": 
			  		lambda m: sum(m.governments[REGION].export_demands.values()),
			  "Bailout cost":
			  		lambda m: m.governments[REGION].bailout_cost,
			  "New firms resources":
			  		lambda m: m.governments[REGION].new_firms_resources,
			  "Unemployment rate":
			  		lambda m: m.governments[REGION].unemployment_rate,
			  "Min wage":
			  		lambda m: m.governments[REGION].min_wage,
			  "Avg wage":
			  		lambda m: m.governments[REGION].avg_wage,
			  }

agent_vars = {"Type":
					lambda a: type(a),
			  "Price":
				  	lambda a: getattr(a, "price", None),
			  "Market share":
			  		lambda a: getattr(a, "market_share", None)[0]
			  				  if getattr(a, "market_share", None) is not None
			  				  else None,
			  "Prod":
			  		lambda a: getattr(a, "prod", None)
			  				  if getattr(a, "prod", None) is not None
			  				  else None,
			  "Real demand":
			  		lambda a: getattr(a, "real_demand", None),
			  "Wage":
			  		lambda a: getattr(a, "wage", None),
			  "Net worth":
			  		lambda a: getattr(a, "net_worth", None),
			  "Size":
			  		lambda a: getattr(a, "size", None),
			  "Labor demand":
			  		lambda a: getattr(a, "labor_demand", None),
			  }