from typing import Dict

import math
import numpy as np
import pandas as pd

from FoodBank import FoodBank
from constants import *
from utils import Food, mod_beta_random

rng = np.random.default_rng()


class FoodPantry:
    def __init__(self, parent: FoodBank, households=100):
        self.parent = parent
        self.households = households
        self.food = Food()
        self.clients = self.generate_clients()
        self.operation_day = rng.integers(0, 7)
        self.previous_record = None

    def generate_clients(self) -> pd.DataFrame:
        columns = [("num_people", ""), (STP, "total"), (STP, "base_secured"), (STP, "secured"), (STP, "demand"),
                   (STP, "purchased"),
                   (FV, "total"), (FV, "base_secured"), (FV, "secured"), (FV, "demand"), (FV, "demand_alt"),
                   (FV, "purchased_fresh"), (FV, "purchased_packaged"),
                   (PT, "total"), (PT, "base_secured"), (PT, "secured"), (PT, "demand"), (PT, "demand_alt"),
                   (PT, "purchased_fresh"), (PT, "purchased_packaged")]
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        clients[("num_people", "")] = rng.choice(range(1, 11), self.households, p=FAMILY_DISTRIBUTION)
        clients.loc[:, (slice(None), "base_secured")] = rng.uniform(0.3, 0.9, (self.households, 3))
        for tp, value in PERSON_WEEKLY_DEMAND.items():
            mean, std = value["mean"], value["std"]
            low, high = 0.5 * mean, 2 * mean
            clients[(tp, "total")] = mod_beta_random(low, high, mean, std, self.households) * clients[
                ("num_people", "")]
        return clients

    def initialize_weekly_demand(self):
        # price_ratio = Food.price.ratio()
        price_ratio = {STP: 1.1, FV: 1.2, PT: 0.9}
        factor = {k: (v ** ELASTICITY[k]) for k, v in price_ratio.items()}
        for tp in ELASTICITY.keys():
            self.clients[(tp, "secured")] = self.clients[(tp, "base_secured")] * factor[tp]
            self.clients[(tp, "demand")] = self.clients[(tp, "total")] * (
                    1 - self.clients[(tp, "secured")]) + rng.normal(0, 100, self.households)
            if "demand_alt" in self.clients[tp]:
                self.clients[(tp, "demand_alt")] = 0.
        # remove the purchase record of the previous week
        self.clients.loc[:, (slice(None), ['purchased', 'purchased_fresh', 'purchased_packaged'])] = 0.

    def run_one_day(self):
        if (Global.get_day() % 7) != self.operation_day:
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
        # TO DO: test the method, and make it more readable
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
        # To do: set limit on fresh food based on the gap between est_demand and order
        return order

    def allocate_food(self, tp, dmd_col, pcs_col) -> pd.DataFrame:
        """
        Clients line up to purchase one type of food. Record their purchase and update the pantry inventory.
        :param tp: the type of food
        :param dmd_col: the name of the column from which we read the client demand
        :param pcs_col: The name of the column where we record the purchases of clients
        :return:
        """
        type_df = self.food.df[self.food.df["type"] == tp].reset_index(drop=True)
        cum_stock = type_df["quantity"].cumsum()
        stock = cum_stock.iat[-1]
        cum_demand = self.clients[dmd_col].cumsum()
        demand = cum_demand.iat[-1]
        if stock >= demand:
            # Get the index of the last batch of food before all demand is satisfied
            pivot = cum_stock.ge(demand).idxmax()
            type_df.loc[pivot, "quantity"] = cum_stock[pivot] - demand
            type_df = type_df[pivot:]
            self.clients.loc[:, pcs_col] = self.clients.loc[:, dmd_col]
        else:
            type_df = pd.DataFrame()
            # Get the index of the first client who cannot get enough food
            pivot = cum_demand.gt(stock).idxmax()
            self.clients.loc[:pivot, pcs_col] = self.clients.loc[:pivot, dmd_col]
            self.clients.loc[pivot, pcs_col] = self.clients.loc[pivot, dmd_col] - (cum_demand[pivot] - stock)
        return type_df

    def hold_pantry(self):
        """ Hold a pantry event. Although in reality one client shops multiple types of food at once, to avoid
        unnecessary computation, we transform it to the equivalent process of allocating food multiple times, once for
        each type of food.
        """
        limit = 20

        types = {STP: [STP], FV: [FFV, PFV], PT: [FPT, PPT]}
        remains = []
        for tp, options in types.items():
            if len(options) == 1:
                remains.append(self.allocate_food(tp, (tp, "demand"), (tp, "purchased")))
            elif len(options) == 2:
                priority, alt = options
                # Transfer out-of-limit demand for fresh food to packaged food
                self.clients.loc[self.clients[(tp, "demand")] > limit, "demand_alt"] = self.clients.loc[self.clients[(
                    tp, "demand")] > limit, "demand"] - limit
                self.clients.loc[self.clients[(tp, "demand")] > limit, "demand"] = limit
                remains.append(self.allocate_food(priority, (tp, "demand"), (tp, "purchased_fresh")))
                # After allocating fresh food, transfer the unmet demand to packaged food
                self.clients[(tp, "demand_alt")] += (
                        self.clients[(tp, "demand")] - self.clients[(tp, "purchased_fresh")])
                remains.append(self.allocate_food(alt, (tp, "demand_alt"), (tp, "purchased_packaged")))
        self.food.df = pd.concat(remains).reset_index(drop=True)

    def func(self, data: pd.Series, typ="exp", param=0.7) -> pd.Series:
        """
        Food utility as a function of the proportion of demand satisfied. It should map 0 to 1, 1 to 1, and be concave
        to reflect diminished marginal effect.
        :param data: a pd.Series or np.ndarray storing the proportion of food demand that is satisfied per household
        :param typ: the type of the function, either exponential, logarithm or quadratic.
        :param param: one additional parameter to decide the shape of the curve, e.g. the power in exponential function,
        the quadracitc coefficient in a quadratic function.
        :return: The element-wise utility value
        >>> pantry = FoodPantry(None)
        >>> portion = pd.Series(np.arange(0, 1.2, 0.2))
        >>> portion.round(2).values.tolist()
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> pantry.func(portion).round(2).values.tolist()
        [0.0, 0.32, 0.53, 0.7, 0.86, 1.0]
        >>> pantry.func(portion, "quad", -0.5).round(2).values.tolist()
        [0.0, 0.28, 0.52, 0.72, 0.88, 1.0]
        >>> pantry.func(portion, "log", 3).round(2).values.tolist()
        [0.0, 0.34, 0.57, 0.74, 0.88, 1.0]
            """
        assert typ in ["exp", "log", "quad"]
        if typ == "exp":
            return np.power(data, param)
        elif typ == "log":
            return np.log(data * param + 1) / math.log(param + 1)
        elif typ == "quad":
            assert -1 <= param < 0, "The quadratic coefficient should be between -1 and 0 for quadratic functions!"
            return param * np.square(data) + (1 - param) * data

    def utility_per_type(self, typ: str) -> float:
        """After a pantry activity, estimate the increment in the utility of one type of food per household.
        :param typ: The type of food
        :return:
        """
        assert typ in [STP, FV, PT]
        get_utility = []
        family_size = self.clients[("num_people", "")]
        total = self.clients[(typ, "total")]
        secured = self.clients[(typ, "secured")]
        if typ == STP:
            delta = self.clients[(STP, "purchased")] / total
            inc = (self.func(secured + delta) - self.func(secured)) * family_size
        else:
            portion1 = self.clients[(typ, "purchased_fresh")] / self.clients[(typ, "total")]
            portion2 = self.clients[(typ, "purchased_packaged")] / self.clients[(typ, "total")]
            cum1 = portion1 + secured
            cum2 = cum1 + portion2
            inc = (0.7 * self.func(cum2) + 0.3 * self.func(cum1) - self.func(secured)) * family_size
        return inc

    def get_utility(self) -> float:
        """Estimate the increment in total food utility after a pantry activity
        :return:
        """
        num_clients = self.clients[("num_people", "")].sum()
        tot_util = pd.Series(np.zeros(self.households))
        for typ in [STP, FV, PT]:
            tot_util += self.utility_per_type(typ)
        return tot_util.sum() / num_clients

