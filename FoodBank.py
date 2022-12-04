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
    ppl_per_pantry = 245
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
    new_food = Food.generate_donation(food_donations)
    self._storage.add(new_food)
    for pantry in self.pantries:
      waste, order, utility = pantry.run_one_day()
      self.total_utility = FoodBank.increment_var(self.total_utility, utility)
      self.total_waste = FoodBank.increment_var(self.total_waste, waste)
      self.fulfill_order(order, pantry)

  def fulfill_order(self, order: Dict(str, float), pantry):
    """Fulfills pantry order

    :param order: order dictionary in weight units
    :param pantry: pantry that made the order
    """
    pass

  def get_pantry_demant_proportion(self):
    """Returns demand in proportions. Used to decide what food to buy next.

    :return: demand proportions
    """
    total = sum(self.pantry_demand.values())
    return { food: (amount/total) for (food, amount) in self.pantry_demand.items() }

  def update_demand(self, order):
    """Updates pantry demand values

    :param order: order made by a pantry
    """
    for food, amount in order.items():
      self.pantry_demand[food] += amount

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
