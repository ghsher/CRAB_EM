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
			"n_agr_firms": 
			  		lambda m: len(m.get_firms_by_type(Agriculture, REGION)),
			"n_ind_firms": 
			  		lambda m: len(m.get_firms_by_type(Industry, REGION)),
            "n_cons_firms": 
		  			lambda m: len(m.get_firms_by_type(Construction, REGION)),
			"n_trans_firms": 
			  		lambda m: len(m.get_firms_by_type(Transport, REGION)),
			"n_inf_firms": 
			  		lambda m: len(m.get_firms_by_type(Information, REGION)),
			"n_fin_firms": 
			  		lambda m: len(m.get_firms_by_type(Finance, REGION)),
			"n_rec_firms": 
			  		lambda m: len(m.get_firms_by_type(Recreation, REGION)),
                
			"HH consumption": 
			  		lambda m: sum(hh.consumption for hh in m.get_households(REGION)),
			"Regional demand":
			  		lambda m: sum(m.governments[REGION].regional_demands.values()),
			"Export demand": 
			  		lambda m: sum(m.governments[REGION].export_demands.values()),
			  # "Bailout cost":
			  # 		lambda m: m.governments[REGION].bailout_cost,
			  # "New firms resources":
			  # 		lambda m: m.governments[REGION].new_firms_resources,
			"Unemployment rate":
			  		lambda m: m.governments[REGION].unemployment_rate,
			"Min wage":
			  		lambda m: m.governments[REGION].min_wage,
			"Avg wage":
			  		lambda m: m.governments[REGION].avg_wage,
	  		"Frac machines dead":
	  		  		lambda m: m.machine_dead/sum(len(firm.capital_vintage)
                                    			 for firm in m.get_firms(0)),
	  		"Changed supplier cap":
	  		  		lambda m: m.changed_supplier[CapitalFirm],
  		  	"Changed supplier agr":
	  		  		lambda m: m.changed_supplier[Agriculture],
	  		"Changed supplier ind":
	  		  		lambda m: m.changed_supplier[Industry],
			"Changed supplier cons":
					lambda m: m.changed_supplier[Construction],
			"Changed supplier trans":
					lambda m: m.changed_supplier[Transport],
			"Changed supplier inf":
					lambda m: m.changed_supplier[Information],
			"Changed supplier fin":
					lambda m: m.changed_supplier[Finance],
			"Changed supplier rec":		
					lambda m: m.changed_supplier[Recreation]
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
			  "Production made":
			  		lambda a: getattr(a, "production_made", None),
			  # "Past demand":
			  # 		lambda a: getattr(a, "past_demand", None)[-1]
			  # 				  if getattr(a, "past_demand", None) is not None
			  # 				  else None,
			  "Real demand":
			  		lambda a: getattr(a, "real_demand", None),
			  "Wage":
			  		lambda a: getattr(a, "wage", None),
			  "Net worth":
			  		lambda a: getattr(a, "net_worth", None),
			  "Size":
			  		lambda a: getattr(a, "size", None),
			  "Labor demand":
			  		lambda a: getattr(a, "desired_employees", None),
			  "Capital desired":
			  		lambda a: getattr(a, "n_desired", None),
			  "Capital amount":
			  		lambda a: sum(vin.amount
			  					  for vin in getattr(a, "capital_vintage", None))
			  				  if getattr(a, "capital_vintage", None) is not None
			  				  else None,
			  "Capital ordered":
			  		lambda a: getattr(a, "n_ordered", None)
			  }