"""
Design of classes
"""
from typing import List

import networkx as nx
import pandas as pd


class Calendar:
    def __init__(self, day=0):
        self.day = None
        self.week = None
        self.day_of_week = None

    def update(self, num_days=1):
        pass

    def revert(self, num_days=1) -> Calendar:
        pass


class Food:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def quality_control(self, num_days=1):
        pass

    def add(self, other):
        pass

    def substract(self, other):
        pass


class FoodPantry:
    def __init__(self, parent: FoodBank):
        self.parent = parent
        self.clients = self.generate_clients(num_household=100)
        self.food = self.initialize_food()
        self.calendar = Calendar()
        self.operation_days = self.generate_op_days()
        # To maximize freshness, the ordered food arrives the day before each pantry event. Assume that the order is
        # made on the same day so that it is based on the latest information about the food in stock.
        self.order_day = max(min(self.operation_days) - 1, 0)
        self.record = None

    def run_one_day(self):
        self.calendar.update()
        if self.calendar.day_of_week = 0:
            self.record = new_week_record()
        # Before a pantry is possibly held, throw away out-of-date food and record the food waste
        waste = self.food.quality_control()
        order = None
        if self.calendar.day_of_week in self.order_day:
            order = make_order(self.parent.food, demand)
        if self.calendar.day_of_week in self.operation_days:
            pass  # hold a pantry event
        if self.calendar.day_of_week == 6:
            pass
        return waste, order

    def make_order(self, food, demand):
        pass

    def new_week_record(self):
        pass
