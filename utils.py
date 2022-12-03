from typing import Dict

import numpy as np
import pandas as pd

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
    beta = np.random.default_rng().beta(a, b, samples)
    beta = low + beta * (high - low)
    return beta


class Food:
<<<<<<< HEAD
    def __init__(self, inventory=None):
=======
    def __init__(self, stock=None):
>>>>>>> e317a5faca45b3a41b95fd94f637ae801d0be7d9
        """
        Initialize a Food object which is either empty, or based on a dataframe or total pounds of food.
        :param stock:
        >>> Food().df
        Empty DataFrame
        Columns: [type, remaining_days, quantity]
        Index: []
        >>> a = Food(5000).df
        >>> a.round(2)  # doctest: +ELLIPSIS
                         type  remaining_days  quantity
        0             staples               1      8.33
        1             staples               2      8.33
        2             staples               3      8.33
        ...
        741  packaged_protein             178      6.94
        742  packaged_protein             179      6.94
        743  packaged_protein             180      6.94
        <BLANKLINE>
        [744 rows x 3 columns]
        >>> a.equals(Food(a).df)
        True
        """
        if stock is None:
            self.df = pd.DataFrame(columns=[
                "type",
                "remaining_days",
                "quantity"
            ]).astype(dtype={
                "type": str,
                "remaining_days": int,
                "quantity": float
            })
        elif isinstance(stock, pd.DataFrame):
            self.df = stock
        elif isinstance(stock, (float, int)):
            assert stock > 0
            types = []
            remaining_days = []
            quantity = []
            for t in TYPES.keys():
                # Assume that the remaining shelf lives of foods are uniformly distributed within [1, max_days]
                max_days = TYPES[t]["max_days"]
                q = stock * TYPES[t]["proportion"] / max_days
                types.extend([t] * max_days)
                remaining_days.extend(list(range(1, max_days + 1)))
                quantity.extend([q] * max_days)
            self.df = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        else:
            raise ValueError("Invalid input for initialization")

    @classmethod
    def generate_donation(cls, mean_total: float):
        """Generate donated food to a food bank in a day. The quantity of different types and the total are all random, but
        their mean values are derived from anual statistics.
        :param mean_total: the mean of the total pounds of foods donated to a food bank per day.
        :return:
        >>> food = Food.generate_donation(5000).df
        """
        types = []
        quantity = []
        remaining_days = []
        for t in TYPES.keys():
            mean = mean_total * TYPES[t]["proportion"]
            low, high, stdev = 0.3 * mean, 5 * mean, 0.5 * mean
            beta = mod_beta_random(low, high, mean, stdev, 1).item()
            # Assume that the remaining shelf lives of foods are uniformly distributed within [1, max_days]
            max_days = TYPES[t]["max_days"]
            q = beta / max_days
            types.extend([t] * max_days)
            remaining_days.extend(list(range(1, max_days + 1)))
            quantity.extend([q] * max_days)
        df = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        return Food(df)

    def sort_by_freshness(self, reverse=False, inplace=True):
        """
        Sort the food in each category by the remaining shelf life. Assume that clients prefer the freshest food,
        whereas food bank gives out food that is going to expire in order to reduce waste.
        :return:
        >>> a = Food(5000)
        >>> a.sort_by_freshness()
        >>> a.df.round(2) # doctest: +ELLIPSIS
                                    type  remaining_days  quantity
        0    fresh_fruits_and_vegetables              14     35.71
        1    fresh_fruits_and_vegetables              13     35.71
        2    fresh_fruits_and_vegetables              12     35.71
        ...
        741                      staples               3      8.33
        742                      staples               2      8.33
        743                      staples               1      8.33
        <BLANKLINE>
        [744 rows x 3 columns]
        """
        sorted_df = self.df.sort_values(by=["type", "remaining_days"], ascending=[True, reverse]).reset_index(
            drop=True)
        if not inplace:
            return Food(sorted_df)
        self.df = sorted_df

    def get_quantity(self) -> Dict[str, float]:
        """

        :param data:
        :return:
        """
        return self.df.groupby(["type"])["quantity"].agg("sum").to_dict()

    def quality_control(self, num_days=1) -> Dict[str, float]:
        """ Subtract some days from the remaining shelf life of the food, remove the expired food from stock, and record
        the quantity of waste in each category.
        :param num_days: number of days since the last quality check
        :return: a dictionary storing the wasted food in each category
        >>> a = Food(5000)
        >>> sum(a.quality_control(float("inf")).values()) == 5000
        True
        >>> res = Food(2000).quality_control(10)
        >>> {k: round(v, 2) for k, v in res.items()}  # doctest: +NORMALIZE_WHITESPACE
        {'fresh_fruits_and_vegetables': 142.86, 'fresh_protein': 200.0, 'packaged_fruits_and_vegetables': 13.89,
        'packaged_protein': 27.78, 'staples': 33.33}
        >>> res1 = Food(2000).quality_control(5)
        >>> res2 = Food(2000).quality_control(20)
        >>> all([res1[key] <= value <= res2[key] for key, value in res.items()])
        True
        """
        self.df["remaining_days"] -= num_days
        mask = self.df["remaining_days"] <= 0
        waste = Food(self.df[mask])
        waste_counter = waste.get_quantity()
        self.df = self.df[~mask]
        return waste_counter

    def add(self, other) -> None:
        """ Add a new batch of food to stock. Merge food items with same type and remaining days.
        Fully tested on jupyter notebook. Still thinking of how to present tests concisely in doctrings
        :param other:
        :return:
        """
        if isinstance(other, Food):
            other = other.df
        self.df = self.df.set_index(["type", "remaining_days"]).add(other.set_index(["type", "remaining_days"]),
                                                                    fill_value=0).reset_index()

    def subtract(self, order: Dict[str, float]):
        """
        Subtract some quantity of food from stock, and return the Food object with that quantity.
        :param order:
        :return:
        >>> food = Food(5000)
        >>> q = food.get_quantity()
        >>> order = {k: v-7 for k, v in q.items()}  # Take away all but 7 pounds in each type
        >>> sent = food.subtract(order).sort_by_freshness(inplace=False)
        >>> food.df.round(3)  # remaining food
                                     type  remaining_days  quantity
        0     fresh_fruits_and_vegetables              14     7.000
        1                   fresh_protein              10     7.000
        2  packaged_fruits_and_vegetables             358     0.056
        3  packaged_fruits_and_vegetables             359     3.472
        4  packaged_fruits_and_vegetables             360     3.472
        5                packaged_protein             179     0.056
        6                packaged_protein             180     6.944
        7                         staples             180     7.000
        >>> sent.df[sent.df["type"] == STP].round(3)  # doctest: +ELLIPSIS
                type  remaining_days  quantity
        561  staples             180     1.333
        562  staples             179     8.333
        563  staples             178     8.333
        ...
        <BLANKLINE>
        [180 rows x 3 columns]
        >>> sent.df[sent.df["type"] == PPT].round(3)  # doctest: +ELLIPSIS
                         type  remaining_days  quantity
        382  packaged_protein             179     6.889
        383  packaged_protein             178     6.944
        384  packaged_protein             177     6.944
        ...
        <BLANKLINE>
        [179 rows x 3 columns]
        >>> food2 = Food(5000)
        >>> food2.subtract({k: v+7 for k, v in q.items()})
        Traceback (most recent call last):
        ValueError: The "fresh_fruits_and_vegetables" you ordered does not exist or is not sufficient in stock
        >>> food2.df["quantity"].sum() == 5000  # Subtraction failed, stock remains the same
        True
        """
        quantity = self.get_quantity()
        self.sort_by_freshness(reverse=True)
        for tp, demand in order.items():
            if demand <= 0:
                continue
            if (tp not in quantity) or (demand > quantity[tp]):
                raise ValueError(f"The \"{tp}\" you ordered does not exist or is not sufficient in stock")

        order = pd.DataFrame(order.items(), columns=["type", "demand"])
        stock = self.df.copy()
        stock = stock.merge(order, on="type", how="left")
        stock["cum_sum"] = stock.groupby("type")["quantity"].cumsum()
        # Due to float precision, loosen the condition a bit
        stock["satisfied"] = stock["cum_sum"] >= (stock["demand"] - 1e-7)
        # Get the first row in each type where the demand is satisfied
        pivot = stock.groupby("type")["satisfied"].idxmax().reset_index().rename(columns={"satisfied": "pivot"})
        stock = stock.merge(pivot, on="type", how="left")
        # Split into two parts, and divide the quantity of this row
        sent = stock.loc[stock.index <= stock["pivot"]]
        stock = stock.loc[stock.index >= stock["pivot"]]
        stock.loc[stock.index == stock["pivot"], "quantity"] = stock.loc[stock.index == stock["pivot"], "cum_sum"] - \
                                                               stock.loc[
                                                                   stock.index == stock["pivot"], "demand"]
        sent.loc[sent.index == sent["pivot"], "quantity"] -= sent.loc[sent.index == sent["pivot"], "cum_sum"] - \
                                                             sent.loc[sent.index == sent["pivot"], "demand"]
        sent = sent[["type", "remaining_days", "quantity"]].reset_index(drop=True)
        stock = stock[["type", "remaining_days", "quantity"]].reset_index(drop=True)
        self.df = stock
        return Food(sent)
