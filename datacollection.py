# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This file stores the model and agent variables for the datacollection
of the CRAB model.

"""

from CRAB_agents import *

REGION = 0

model_vars = {# -- FLOOD -- #
			  "Flood": "flood_now",
			  # -- AGENT COUNTS -- #
			  "N households": 
			  		lambda m: len(m.get_households(REGION)),
			  "N cap firms": 
		  			lambda m: len(m.get_firms_by_type(CapitalFirm, REGION)),
			  "N cons firms": 
			  		lambda m: len(m.get_firms_by_type(ConsumptionGoodFirm, REGION)),
			  "N serv firms": 
			  		lambda m: len(m.get_firms_by_type(ServiceFirm, REGION)),

			  # -- FIRM ATTRIBUTES -- #
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
			  "N cap bankrupt":
			  		lambda m: sum(1 for firm in m.firms_to_remove[REGION]
			  					  if type(firm) == CapitalFirm),
			  "N cons bankrupt":
			  		lambda m: sum(1 for firm in m.firms_to_remove[REGION]
			  					  if type(firm) == ConsumptionGoodFirm),
			  "N serv bankrupt":
			  		lambda m: sum(1 for firm in m.firms_to_remove[REGION]
			  					  if type(firm) == ServiceFirm),

			  # -- HOUSEHOLD ATTRIBUTES -- #
			  "Total HH consumption": 
			  		lambda m: sum(hh.consumption for hh in m.get_households(REGION)),
			  "Total HH net worth":
			  		lambda m: sum(hh.net_worth for hh in m.get_households(REGION)),
			  "Total flood damage":
			   		lambda m: sum(hh.monetary_damage
			   					  for hh in m.get_households(REGION)),
  			  # "Adaptation: elevation":
			  # 		lambda m: sum(bool(hh.adaptation["Elevation"])
			  # 					  for hh in m.get_households(REGION)) /
			  # 					  len(m.get_households(REGION)),
			  # "Adaptation: dry-proofing":
			  # 		lambda m: sum(bool(hh.adaptation["Wet_proof"])
			  # 					  for hh in m.get_households(REGION)) /
			  # 					  len(m.get_households(REGION)),
			  # "Adaptation: wet-proofing":
			  # 		lambda m: sum(bool(hh.adaptation["Dry_proof"])
			  # 					  for hh in m.get_households(REGION)) /
			  # 					  len(m.get_households(REGION)),
			  }

agent_vars = {
			  "Type":
			  		lambda a: type(a),
			  "Net worth":
			  		lambda a: getattr(a, "net_worth", None),

			  # -- FIRM ATTRIBUTES -- #
			  "Sales":
			  		lambda a: getattr(a, "sales", None),
			  "Price":
			  	  	lambda a: getattr(a, "price", None),
			  # "Market share":
			  # 		lambda a: getattr(a, "market_share", None)[0]
			  # 				  if getattr(a, "market_share", None) is not None
			  # 				  else None,
			  "Prod":
			  		lambda a: getattr(a, "prod", None),
			  "Machine prod":
			  		lambda a: getattr(a, "machine_prod", None),
			  "Inventories":
			  		lambda a: getattr(a, "inventories", None),
			  # "N ordered":
			  # 		lambda a: getattr(a, "quantity_ordered", None),
			  "Past demand":
			  		lambda a: getattr(a, "past_demand", None)[-1]
			  				  if getattr(a, "past_demand", None) is not None
			  				  else None,
			  "Real demand":
			  		lambda a: getattr(a, "real_demand", None),
			  # "Demand filled":
			  # 		lambda a: getattr(a, "demand_filled", None),
		  	  # "Demand unfilled":
			  # 		lambda a: getattr(a, "unfilled_demand", None),
			  "Wage":
			  		lambda a: getattr(a, "wage", None),
			  # "Size":
			  # 		lambda a: getattr(a, "size", None),
			  "Labor demand":
			  		lambda a: getattr(a, "desired_employees", None),
			  "Capital amount":
			  		lambda a: sum(vin.amount
			  					  for vin in getattr(a, "capital_vintage", None))
			  				  if getattr(a, "capital_vintage", None) is not None
			  				  else None,
			   "Flood damage": "damage_coef",
			  }