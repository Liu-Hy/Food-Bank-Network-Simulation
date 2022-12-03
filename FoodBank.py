import FoodPantry
import pandas as pd
from utils import Food
from typing import List, Dict
from constants import Global


class FoodBank:
  def __init__(self, food_insecure_pop: int, initial_storage: float):
    """Food bank constructor

    :param food_insecure_pop: Number of food insecure people. Used to estimate number of pantries
    :param initial_storage: Initial storage of food in pounds. Value given to Food class
    """
    # we estimated this number from real data of the Eastern Food Bank
    ppl_per_pantry =  260
    # we assume half of the food insecure people actually use the bank
    num_pantries = int(.5 * food_insecure_pop / ppl_per_pantry)
    self.pantries:List('FoodPantry') = [FoodPantry(self) for _ in range(num_pantries)]
    self.total_utility = None
    self.total_waste = None

    food_types = Global.get_food_types()
    self.pantry_demand = dict(zip(food_types, [0] * len(food_types)))

    self._storage = Food(initial_storage)

  def food_storage(self):
    return self._storage
  
  def run_one_day(self, budget: float, food_donations):
    """Runs simulation for the day. Also calls `run_one_day` for each pantry it serves.

    :param budget: Budget for the day
    :param food_donations: Budget for the day
    """
    for pantry in self.pantries:
      waste, order, utility = pantry.run_one_day()
      self.total_utility = FoodBank.increment_var(self.total_utility, utility)
      self.total_waste = FoodBank.increment_var(self.total_waste, waste)

  @classmethod
  def increment_var(self, var: float, increment: float):
    if var is None:
      return increment
    else:
      return var + increment
  
  def buy_food(self, donations: float):
    """Buys food based on donations and demand from pantries.

    :param donations: donations in dollars
    """
    # predicted supply and predicted demand
    # open to receiving food
    # food available to donate
    # feeding america tax
