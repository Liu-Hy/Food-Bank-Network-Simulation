import numpy as np
import pandas as pd
from typing import List, Dict
from constants import *
from FoodBank import FoodBank
from utils import Calendar, Food, mod_beta_random

rng = np.random.default_rng()

class FoodPantry:
    def __init__(self, parent: FoodBank, households=100):
        self.parent = parent
        self.households = households
        self.calendar = Calendar()
        self.food = Food()
        self.clients = self.generate_clients()
        self.base_secured_rate = self.generate_base_secured()
        self.operation_day = rng.integers(0, 7)
        self.previous_record = None

    def generate_clients(self):
        columns = [("num_people", ""), (STP, "demand"), (STP, "secured"), (STP, "lacking"), (STP, "purchased"),
                   (FV, "demand"), (FV, "secured"), (FV, "lacking"), (FV, "purchased_fresh"),
                   (FV, "purchased_packaged"),
                   (PT, "demand"), (PT, "secured"), (PT, "lacking"), (PT, "purchased_fresh"),
                   (PT, "purchased_packaged")]
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        clients[("num_people", "")] = rng.choice(range(1, 11), self.households, p=FAMILY_DISTRIBUTION)
        for type, value in PERSON_WEEKLY_DEMAND.items():
            mean, std = value["mean"], value["std"]
            low, high = 0.5 * mean, 2 * mean
            clients[(type, "demand")] = mod_beta_random(low, high, mean, std, self.households) * clients[("num_people", "")]
        return clients

    def generate_base_secured(self):
        rates = rng.uniform(0.3, 0.8, (self.households, 3))
        return pd.DataFrame(rates, columns=[STP, FV, PT])

    def initialize_weekly_demand(self):
        # price_ratio = Food.price.ratio()
        price_ratio = {STP: 1.1, FV: 1.2, PT: 0.9}
        factor = {k: (v ** ELASTICITY[k]) for k, v in price_ratio.items()}
        for type in ELASTICITY.keys():
            self.clients[(type, "secured")] = self.base_secured_rate[type] * factor[type]
            self.clients[(type, "lacking")] = self.clients[(type, "demand")] * (1 - self.clients[(type, "secured")]) + rng.normal(0, 100, self.households)
        self.clients.loc[:, (slice(None), ['purchased', 'purchased_fresh', 'purchased_packaged'])] = 0.

    def run_one_day(self):
        self.calendar.update()
        if self.calendar.day_of_week != self.operation_day:
            return
        est_demand = self.estimate_demand()
        order = self.make_order(est_demand)
        # Client needs and queuing order changed
        self.initialize_weekly_demand()
        self.clients = self.clients.sample(frac=1).reset_index(drop=True)
        # Prepare food
        waste = self.food.quality_control(num_days=7)
        self.food.sort_by_freshness()
        # TO DO: hold a pantry. Use aggregation instead of loops whenever possible
        inc = self.get_utility()
        return waste, order, inc

    def estimate_demand(self) -> Dict[str, float]:
        """Estimate client demand this week based on prior experience"""
        if self.previous_record is None:
            num_clients = self.clients[("num_people", "")].sum()
            est_demand = {k: (v * num_clients) for k, v in PERSON_WEEKLY_DEMAND.items()}
        else:
            NotImplemented

    def make_order(self, est_demand: Dict[str, float]):
        """ Make an order to the food bank subject to estimated demand and the current inventory.
        Request fresh food when it is available, otherwise request packaged food to fill the demand. """
        order = dict()
        stock = self.parent.df.groupby(["type"])["quantity"].agg("sum").to_dict()
        order[STP] = min(est_demand[STP], stock[STP])
        order[FFV] = min(est_demand[FV], stock[FFV])
        order[PFV] = min(est_demand[FV] - order[FFV], stock[PFV])
        order[FPT] = min(est_demand[PT], stock[FPT])
        order[PFV] = min(est_demand[PT] - order[FPT], stock[PFV])
        return order

    def get_utility(self) -> float:
        """Estimate the increment in social welfare after a pantry event
        :return:
        """
