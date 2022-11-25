import pandas as pd

class Calendar:
    def __init__(self, day=0):
        self.day = day
        self.week = day // 7
        self.day_of_week = day % 7
    def update(self, num_days=1):
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

    def sort_by_freshness(self):
        pass