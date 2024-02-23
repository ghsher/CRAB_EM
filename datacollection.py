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
			  		lambda a: getattr(a, "prod", None),
			  "Inventories":
			  		lambda a: getattr(a, "inventories", None),
			  "N ordered":
			  		lambda a: getattr(a, "quantity_ordered", None),
			  "Production made":
			  		lambda a: getattr(a, "production_made", None),
			  "Sum past demand":
			  		lambda a: sum(getattr(a, "past_demand", None))
			  				  if getattr(a, "past_demand", None) is not None
			  				  else None,
			  "Past demand":
			  		lambda a: getattr(a, "past_demand", None)[-1]
			  				  if getattr(a, "past_demand", None) is not None
			  				  else None,
			  "Real demand":
			  		lambda a: getattr(a, "real_demand", None),
			  "Wage":
			  		lambda a: getattr(a, "wage", None),
			  "Net worth":
			  		lambda a: getattr(a, "net_worth", None),
			  "Debt":
			  		lambda a: getattr(a, "debt", None),
			  "Size":
			  		lambda a: getattr(a, "size", None),
			  "Labor demand":
			  		lambda a: getattr(a, "desired_employees", None),
			  "Capital amount":
			  		lambda a: sum(vin.amount
			  					  for vin in getattr(a, "capital_vintage", None))
			  				  if getattr(a, "capital_vintage", None) is not None
			  				  else None,
			  }