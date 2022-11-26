import pandas as pd

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

    def quality_control(self, num_days=1):
        self.df["remaining_days"] -= num_days
        mask = self.df["remaining_days"] <= 0
        waste = self.df[mask]
        self.df = self.df[~mask]
        waste_counter = waste.groupby(["type"])["quantity"].agg("sum")["quantity"].to_dict()
        return waste_counter

    def add(self, other):
        pass

    def subtract(self, other):
        pass

    def sort_by_freshness(self):
        pass