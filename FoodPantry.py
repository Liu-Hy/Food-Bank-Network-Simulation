import pandas as pd
from typing import List
from utils import Calendar, Food


class FoodPantry:
    def __init__(self, parent: FoodBank):
        self.parent = parent
        self.clients = self.generate_clients(num_household=100)
        self.food = self.initialize_food()
        self.calendar = Calendar()
        self.operation_days = self.generate_op_days()
        self.order_day = max(min(self.operation_days) - 1, 0)
        self.record = None

    def run_one_day(self):
        self.calendar.update()
        if self.calendar.day_of_week == 0:
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
