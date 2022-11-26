import numpy as np
import pandas as pd
from typing import Dict

def mod_beta_random(min: float, max: float, mean: float, std: float, samples: int) -> np.ndarray:
    """
    Generate random numbers from a transformed Beta distribution
    Formulas to derive parameters are taken from here
    https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
    :param min: the lowest value possible
    :param max: the highest value possible
    :param mean: the mean of the population
    :param std: the standard deviation of the population
    :param samples: the number of samples to generate, or a tuple of the output shape
    :return: a np.ndarray of size "shape" drawn from the distribution
    >>> beta = mod_beta_random(0., 10., 4., 1.9, 200)
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
    assert min <= mean <= max, "Population mean out of bound!"
    mu = (mean - min) / (max - min)
    sigma = std / (max - min)
    assert sigma ** 2 <= mu * (1 - mu), "Population standard deviation too large for a Beta distribution to exist."
    a = ((1 - mu) / (sigma ** 2) - 1 / mu) * (mu ** 2)
    b = a * (1 / mu - 1)
    beta = np.random.beta(a, b, samples)
    beta = min + beta * (max - min)
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
    def __init__(self, df: pd.DataFrame=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(columns=[
                "type",
                "quantity",
                "remaining_days"
            ]).astype(dtype={
                "type": str,
                "quantity": float,
                "remaining_days": int,
            })
            self.df = self.df.set_index(["type", "remaining_days"]) # later will try MultiIndex for efficiency

    def initialize_inventory(self, total):
        """Initialize the inventory of a food bank or a pantry
        :return:
        """
        pass

    def generate_donation(self, exp_total):
        """Generate food donation in a day. The quantity of each type and the total are all random, but their means are given
        :return:
        """
        pass

    def sort_by_freshness(self, ascending: bool):
        """
        Sort the food in each category by the remaining shelf life. Assume that clients prefer the freshest food,
        whereas food bank gives out food that is going to expire in order to reduce waste.
        :return:
        """
        self.df = self.df.sort_index(ascending=[True, ascending])

    def quality_control(self, num_days=1) -> Dict[str, float]:
        """ Subtract some days from the remaining shelf life of the food, remove the expired food from the inventory,
        and record the quantity of waste in each category.
        :param num_days: number of days since the last quality check
        :return: a dictionary storing the wasted food in each category
        """
        # Since MultiIndex is immutable, this is the fastest way to my knowledge
        self.df.index = self.df.index.set_levels(self.df.index.levels[1] - num_days, level=1)
        # To Do: test the below lines for multi-indexed dataframe
        mask = self.df["remaining_days"] <= 0
        waste = self.df[mask]
        waste_counter = waste.groupby(level=0)["quantity"].agg("sum")["quantity"].to_dict()
        self.df = self.df[~mask]
        return waste_counter

    def add(self, other):
        """ Add a new batch of food to inventory
        :param other:
        :return:
        """
        if isinstance(other, Food):
            other = other.df
        self.df = self.df.add(other, fill_value=0)
        pass

    def subtract(self, other):
        """
        Subtract an existing batch of food from inventory
        :param other:
        :return:
        """
        if isinstance(other, Food):
            other = other.df
        self.df = self.df.sub(other, fill_value=0)
        # if not valid: raise ValueError("Some food items does not exist or are not enough")

