# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""
@author: TabernaA

StagedActivationByType class.
This class is based on the StagedActivation scheduler from the MESA library,
but with random activation per agent class: each group of agents is shuffled
before activation. In addition, households are sorted by income, to represent
the better opportunities that higher-income households have on the labor market.

"""

from collections import defaultdict
from mesa.time import StagedActivation

from CRAB_agents import *
from government import Government


class StagedActivationByType(StagedActivation):
    """A custom activation class"""

    def __init__(self, model, stage_list, shuffle=True,
                 shuffle_between_stages=False):
        """Initialization of StagedActivationByType class.

        Args:
            model                   : Model object of CRAB_model class
            stage_list              : List of stage names
            shuffle                 : Boolean; if True, shuffle order of agents
                                      each step
            shuffle_between_stages  : Boolean; if True, shuffle agents after
                                      each stage
        """
        super().__init__(model, stage_list, shuffle, shuffle_between_stages)
        self.agents_by_type = defaultdict(list)

    def add(self, agent):
        """Add new agent to schedule. """
        super().add(agent)
        if agent not in self.agents_by_type[type(agent)]:
            self.agents_by_type[type(agent)].append(agent)
        else:
            raise ValueError("agent already added to agent dict")

    def remove(self, agent):
        """Remove agent from schedule. """
        super().remove(agent)
        self.agents_by_type[type(agent)].remove(agent)

    def step(self):
        """Single model step.

        Each step consists of several stages to provide the right order
        of events and dynamics. The stages are executed consecutively,
        so that all agents execute one stage before moving to the next.
        """
        for stage in self.stage_list:
            for agent_type in self.agents_by_type.keys():
                # Create a copy to not change the original list of agents
                agents = self.agents_by_type[agent_type].copy()
                # Shuffle agents per type (except for Government)
                if agent_type != Government:
                    self.model.RNGs[agent_type].shuffle(agents)
                    # Sort households by education (highest education first)
                    if agent_type == Household:
                        agents = sorted(agents, key=lambda hh: hh.education,
                                        reverse=True)
                for agent in agents:
                    getattr(agent, stage)()
            self.time += self.stage_time
        self.time = int(self.time)
        self.steps += 1
