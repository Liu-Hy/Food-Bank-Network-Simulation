import numpy as np
import pandas as pd
from typing import List
from constants import *
from utils import Calendar, Food, mod_beta_random


class FoodPantry:
    def __init__(self, parent: FoodBank, households=100):
        self.parent = parant
        self.households = households
        self.calendar = Calendar()
        self.clients = self.generate_clients()
        self.base_secured_rate = self.generate_base_secured()
        self.operation_day =

    def generate_clients(self):
        columns = [("num_people", ""), (STP, "demand"), (STP, "secured"), (STP, "lacking"), (STP, "purchased"),
                   (FV, "demand"), (FV, "secured"), (FV, "lacking"), (FV, "purchased_fresh"),
                   (FV, "purchased_packaged"),
                   (PT, "demand"), (PT, "secured"), (PT, "lacking"), (PT, "purchased_fresh"),
                   (PT, "purchased_packaged")]
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        clients[("num_people", "")]
        for type, value in PERSON_WEEKLY_DEMAND.items():
            mean, std = value["mean"], value["std"]
            low, high = 0.5 * mean, 2 * mean
            clients[(type, "demand")] = mod_beta_random(low, high, mean, std, self.households)

    def generate_base_secured(self):
        return np.random.default_rng().uniform(0.3, 0.8, (self.households, 3))

    def run_one_day(self):



    def make_order(self, food, demand):
        pass
