from typing import Dict

import numpy as np
import pandas as pd

from FoodBank import FoodBank
from constants import *
from utils import Calendar, Food, mod_beta_random

rng = np.random.default_rng()


class FoodPantry:
    def __init__(self, parent: FoodBank, households=100):
        self.parent = parent
        self.households = households
        self.calendar = Calendar()
        self.food = Food()
        self.clients = self.generate_clients()
        self.base_secure_rate = self.generate_base_secure()
        self.operation_day = rng.integers(0, 7)
        self.previous_record = None

    def generate_clients(self) -> pd.DataFrame:
        columns = [("num_people", ""), (STP, "total"), (STP, "secured"), (STP, "demand"), (STP, "purchased"),
                   (FV, "total"), (FV, "secured"), (FV, "demand"), (FV, "demand_alt"), (FV, "purchased_fresh"),
                   (FV, "purchased_packaged"),
                   (PT, "total"), (PT, "secured"), (PT, "demand"), (PT, "demand_alt"), (PT, "purchased_fresh"),
                   (PT, "purchased_packaged")]
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        clients[("num_people", "")] = rng.choice(range(1, 11), self.households, p=FAMILY_DISTRIBUTION)
        for tp, value in PERSON_WEEKLY_DEMAND.items():
            mean, std = value["mean"], value["std"]
            low, high = 0.5 * mean, 2 * mean
            clients[(tp, "total")] = mod_beta_random(low, high, mean, std, self.households) * clients[
                ("num_people", "")]
        return clients

    def generate_base_secure(self) -> pd.DataFrame:
        rates = rng.uniform(0.3, 0.8, (self.households, 3))
        return pd.DataFrame(rates, columns=[STP, FV, PT])

    def initialize_weekly_demand(self):
        # price_ratio = Food.price.ratio()
        price_ratio = {STP: 1.1, FV: 1.2, PT: 0.9}
        factor = {k: (v ** ELASTICITY[k]) for k, v in price_ratio.items()}
        for tp in ELASTICITY.keys():
            self.clients[(tp, "secured")] = self.base_secure_rate[tp] * factor[tp]
            self.clients[(tp, "demand")] = self.clients[(tp, "total")] * (
                        1 - self.clients[(tp, "secured")]) + rng.normal(0, 100, self.households)
            if "demand_alt" in self.clients[tp]:
                self.clients[(tp, "demand_alt")] = 0

    def run_one_day(self):
        self.calendar.update()
        if self.calendar.day_of_week != self.operation_day:
            return
        est_demand = self.estimate_demand()
        order = self.make_order(est_demand)
        suppl = self.parent.food.subtract(order)  # maybe done by the FoodBank class
        self.food.add(suppl)
        # Prepare food
        waste = self.food.quality_control(num_days=7)
        self.food.sort_by_freshness()
        # The demand and queuing order of clients change every week
        self.initialize_weekly_demand()
        self.clients = self.clients.sample(frac=1).reset_index(drop=True)
        # TO DO: hold a pantry. Use aggregation instead of loops whenever possible
        self.hold_pantry()

        utility = self.get_utility()
        return waste, order, utility

    def estimate_demand(self) -> Dict[str, float]:
        """Predict client demand this week based on prior experience"""
        if self.previous_record is None:
            num_clients = self.clients[("num_people", "")].sum()
            est_demand = {k: (v * num_clients) for k, v in PERSON_WEEKLY_DEMAND.items()}
        else:
            NotImplemented
        return est_demand

    def make_order(self, est_demand: Dict[str, float]):
        """ Make an order to the food bank subject to estimated demand and the current inventory.
        Request fresh food when it is available, otherwise request packaged food to meet the demand. """
        order = dict()
        stock = self.parent.food.get_quantity()
        order[STP] = min(est_demand[STP], stock[STP])
        order[FFV] = min(est_demand[FV], stock[FFV])
        order[PFV] = min(est_demand[FV] - order[FFV], stock[PFV])
        order[FPT] = min(est_demand[PT], stock[FPT])
        order[PFV] = min(est_demand[PT] - order[FPT], stock[PFV])
        return order

    def hold_pantry(self):
        limit = 200
        # remove the purchase record of the previous week
        self.clients.loc[:, (slice(None), ['purchased', 'purchased_fresh', 'purchased_packaged'])] = 0.

        type_df = self.food.df[self.food.df["type"] == STP].reset_index(drop=True)
        stock = type_df["quantity"].sum()
        demand = self.clients[(STP, "demand")].sum()
        if stock >= demand:
            cum_stock = type_df["quantity"].cumsum()
            pivot = cum_stock.ge(demand).idxmax()
            type_df.loc[pivot, "quantity"] = cum_stock[pivot] - demand
            type_df = type_df[pivot:]
            self.clients.loc[:, (STP, "purchased")] = self.clients.loc[:, (STP, "demand")]
        else:
            type_df = pd.DataFrame()
            cum_demand = self.clients[(STP, "demand")].cumsum()
            pivot = cum_demand.gt(stock).idxmax()
            self.clients.loc[:pivot, (STP, "purchased")] = self.clients.loc[:pivot, (STP, "demand")]
            self.clients.loc[pivot, (STP, "purchased")] = self.clients.loc[pivot, (STP, "demand")] - (cum_demand[pivot] - stock)

        # for fresh fruits and vegetables
        type_df = self.food.df[self.food.df["type"] == FFV].reset_index(drop=True)
        type_df2 = self.food.df[self.food.df["type"] == PFV].reset_index(drop=True)
        self.clients.loc[self.clients[(FV, "demand")] > limit, "demand_alt"] = self.clients.loc[self.clients[(FV, "demand")] > limit, "demand"] - limit
        self.clients.loc[self.clients[(FV, "demand")] > limit, "demand"] = limit
        stock = type_df["quantity"].sum()
        demand = self.clients[(FV, "demand")].sum()
        if stock >= demand:
            cum_stock = type_df["quantity"].cumsum()
            pivot = cum_stock.ge(demand).idxmax()
            type_df.loc[pivot, "quantity"] = cum_stock[pivot] - demand
            type_df = type_df[pivot:]
            self.clients.loc[:, (FV, "purchased_fresh")] = self.clients.loc[:, (FV, "demand")]
        else:
            type_df = pd.DataFrame()
            cum_demand = self.clients[(FV, "demand")].cumsum()
            pivot = cum_demand.gt(stock).idxmax()
            self.clients.loc[:pivot, (FV, "purchased_fresh")] = self.clients.loc[:pivot, (FV, "demand")]
            shortage = cum_demand[pivot] - stock
            self.clients.loc[pivot, (FV, "purchased_fresh")] = self.clients.loc[pivot, (FV, "demand")] - shortage

            # transfer excess demand to packaged food
            self.clients.loc[pivot, (FV, "demand_alt")] += shortage
            self.clients.loc[pivot+1, (FV, "demand_alt")] += self.clients.loc[pivot+1, (FV, "demand")]
            # repeat the process to packaged food
            stock = type_df2["quantity"].sum()
            demand = self.clients[(FV, "demand_alt")].sum()
            if stock >= demand:
                cum_stock = type_df2["quantity"].cumsum()
                pivot = cum_stock.ge(demand).idxmax()
                type_df2.loc[pivot, "quantity"] = cum_stock[pivot] - demand
                type_df2 = type_df2[pivot:]
                self.clients.loc[:, (FV, "purchased_packaged")] = self.clients.loc[:, (STP, "demand_alt")]
            else:
                type_df2 = pd.DataFrame()
                cum_demand = self.clients[(STP, "demand_alt")].cumsum()
                pivot = cum_demand.gt(stock).idxmax()
                self.clients.loc[:pivot, (FV, "purchased_packaged")] = self.clients.loc[:pivot, (FV, "demand_alt")]
                self.clients.loc[pivot, (FV, "purchased_packaged")] = self.clients.loc[pivot, (FV, "demand_alt")] - (
                            cum_demand[pivot] - stock)
        # repeat for proteins

    def get_utility(self) -> float:
        """Estimate the increment in social welfare after a pantry event
        :return:
        """
