import numpy as np
import pandas as pd
from typing import Dict
from constants import *

def mod_beta_random(low: float, high: float, mean: float, std: float, samples: int) -> np.ndarray:
    """
    Generate random numbers from a transformed Beta distribution
    Formulas to derive parameters are taken from here
    https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
    :param low: the lowest value possible
    :param high: the highest value possible
    :param mean: the mean of the population
    :param std: the standard deviation of the population
    :param samples: the number of samples to generate, or a tuple of the output shape
    :return: a np.ndarray of size "shape" drawn from the distribution
    >>> beta = mod_beta_random(0., 10., 4., 1.9, 500)
    >>> np.all((0 <= beta) & (beta <= 10))
    True
    >>> np.all((0.5 <= beta) & (beta <= 9.5))
    False
    >>> mu = beta.mean()
    >>> 3.8 < mu and mu < 4.2
    True
    >>> std = beta.std()
    >>> 1.8 < std and std < 2.0
    True
    """
    assert low <= mean <= high, "Population mean out of bound."
    mu = (mean - low) / (high - low)
    sigma = std / (high - low)
    assert sigma ** 2 <= mu * (1 - mu), "Population standard deviation too large for a Beta distribution to exist."
    a = ((1 - mu) / (sigma ** 2) - 1 / mu) * (mu ** 2)
    b = a * (1 / mu - 1)
    beta = np.random.beta(a, b, samples)
    beta = low + beta * (high - low)
    return beta


class Calendar:
    def __init__(self, day=0):
        self.day = day
        self.week = day // 7
        self.day_of_week = day % 7

    def update(self, num_days=1):
        self.day += num_days
        self.week = self.day // 7
        self.day_of_week = self.day % 7


class Food:
    def __init__(self, inventory=None):
        """
        Initialize a Food object which is either empty, or based on a dataframe or total pounds of food.
        :param inventory:
        >>> Food().df
        Empty DataFrame
        Columns: [type, quantity, remaining_days]
        Index: []
        >>> a = Food(5000).df
        >>> a
                         type  quantity  remaining_days
        0             staples  8.333333               1
        1             staples  8.333333               2
        2             staples  8.333333               3
        3             staples  8.333333               4
        4             staples  8.333333               5
        ..                ...       ...             ...
        739  packaged protein  6.944444             176
        740  packaged protein  6.944444             177
        741  packaged protein  6.944444             178
        742  packaged protein  6.944444             179
        743  packaged protein  6.944444             180
        <BLANKLINE>
        [744 rows x 3 columns]
        >>> a.equals(Food(a).df)
        True
        """
        if inventory is None:
            self.df = pd.DataFrame(columns=[
                "type",
                "quantity",
                "remaining_days"
            ]).astype(dtype={
                "type": str,
                "quantity": float,
                "remaining_days": int,
            })
        elif isinstance(inventory, pd.DataFrame):
            self.df = inventory
        elif isinstance(inventory, (float, int)):
            type = []
            quantity = []
            remaining_days = []
            for t in TYPES.keys():
                # Assume that the remaining shelf lives of foods are uniformly distributed within [1, max_days]
                q = inventory * TYPES[t]["proportion"] / TYPES[t]["max_days"]
                for d in range(1, TYPES[t]["max_days"] + 1):
                    type.append(t)
                    quantity.append(q)
                    remaining_days.append(d)
            self.df = pd.DataFrame({"type": type, "quantity": quantity, "remaining_days": remaining_days})
        else:
            raise ValueError("Invalid input for initialization")
            #self.df = self.df.set_index(["type", "remaining_days"])


    @classmethod
    def generate_donation(cls, mean_total: float):
        """Generate food donation in a day. The quantity of each type and the total are all random, but their means are given
        :return:
        >>> food = Food.generate_donation(5000).df
        """
        type = []
        quantity = []
        remaining_days = []
        for t in TYPES.keys():
            # Assume that the remaining shelf lives of foods are uniformly distributed within [1, max_days]
            mean = mean_total * TYPES[t]["proportion"]
            low, high, stdev = 0.3 * mean, 5 * mean, 0.5 * mean
            beta = mod_beta_random(low, high, mean, stdev, 1).item()
            q = beta / TYPES[t]["max_days"]
            for d in range(1, TYPES[t]["max_days"] + 1):
                type.append(t)
                quantity.append(q)
                remaining_days.append(d)
        df = pd.DataFrame({"type": type, "quantity": quantity, "remaining_days": remaining_days})
        return Food(df)

    def sort_by_freshness(self, ascending=False):
        """
        Sort the food in each category by the remaining shelf life. Assume that clients prefer the freshest food,
        whereas food bank gives out food that is going to expire in order to reduce waste.
        :return:
        """
        self.df = self.df.sort_values(by=["type", "remaining_days"], ascending=[True, ascending])

    def quality_control(self, num_days=1) -> Dict[str, float]:
        """ Subtract some days from the remaining shelf life of the food, remove the expired food from the inventory,
        and record the quantity of waste in each category.
        :param num_days: number of days since the last quality check
        :return: a dictionary storing the wasted food in each category
        """
        self.df["remaining_days"] -= num_days
        mask = self.df["remaining_days"] <= 0
        waste = self.df[mask]
        waste_counter = waste.groupby(["type"])["quantity"].agg("sum")["quantity"].to_dict()
        self.df = self.df[~mask]
        return waste_counter

    def add(self, other):
        """ Add a new batch of food to inventory
        :param other:
        :return:
        >>> food1 = Food.generate_donation(100)
        >>> food1.df
        >>> food1.add(Food.generate_donation(100))
        >>> food1.df
        """
        if isinstance(other, Food):
            other = other.df
        self.df = self.df.set_index(["type", "remaining_days"]).add(other.set_index(["type", "remaining_days"]),
                                                                    fill_value=0).reset_index()

    def subtract(self, other):
        """
        Subtract an existing batch of food from inventory
        :param other:
        :return:
        """
        if isinstance(other, Food):
            other = other.df
        self.df = self.df.set_index(["type", "remaining_days"]).sub(other.set_index(["type", "remaining_days"]),
                                                                    fill_value=0).reset_index()
        # if not valid: raise ValueError("Some food items does not exist or are not enough")

