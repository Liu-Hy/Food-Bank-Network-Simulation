from FoodPantry import *
import pandas as pd
from utils import Food
from typing import List, Dict, Tuple, Any
from Global import Global, TYPES
import cython

@cython.cclass
class FoodBank:
    @cython.ccall
    def __init__(self, food_insecure_pop: int, initial_storage: float,
                 households_per_pantry: int = Global.households_per_pantry):
        """Food bank constructor. Modify `food_insecure_pop` and `households_per_pantry` to simulate crisis.
        :param food_insecure_pop: Number of food insecure people. Used to estimate number of pantries
        :param initial_storage: Initial storage of food in pounds. Value given to Food class
        :param households_per_pantry: default to global number

        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=2_000)
        >>> len(bank.pantries)
        2
        >>> bank.pantry_demand #doctest: +ELLIPSIS
        {'staples': 0, 'fresh_fruits_and_vegetables': 0, ... 'fresh_protein': 0, 'packaged_protein': 0}
        >>> int(sum(bank.get_food_quantity().values()))
        2000
        >>> bank.last_week_demand
        []
        """
        # we assume a quarter of the food insecure people actually use the bank
        num_pantries = int(POPULATION_FACTOR * food_insecure_pop / households_per_pantry)
        self.pantries: List[FoodPantry] = [FoodPantry(self, num_households=households_per_pantry) for _ in
                                           range(num_pantries)]

        food_types = FOOD_GOODS
        self.pantry_demand = dict(zip(food_types, [0] * len(food_types)))

        self.storage = Food(initial_storage)

        self.last_week_demand: List[Dict[str, int]] = []

        self.last_purchase: pd.DataFrame | None = None

        self.last_donation = 0

    @cython.ccall
    def next_week_storage_estimate(self) -> Food:
        """Next week's storage considering last week's demand
        :return: future storage Food instance

        >>> initial_storage = 2000
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> int(sum(bank.next_week_storage_estimate().get_quantity().values()))
        2000
        >>> for i in range(7):
        ...     _ = bank.run_one_day(2_000, 2_000)
        >>> int(sum(bank.next_week_storage_estimate().get_quantity().values())) > initial_storage
        True
        """
        _, future_storage = self.storage.subtract(self.last_week_aggregate_demand(), predict=True)

        if self.last_purchase is not None:
            future_storage.add(self.last_purchase)
            future_storage.add(Food(self.last_donation * 7))

        return future_storage

    @cython.ccall
    def food_going_bad(self) -> Food:
        """Returns Food instance with food going bad considering current demand
        :return: Food instance with all the food that is going bad within next week considering current demand

        >>> initial_storage = 1
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> for i in range(7):
        ...     _ = bank.run_one_day(2, 2)
        >>> sum(bank.food_going_bad().get_quantity().values()) < 10 # should be a small value
        True
        >>> initial_storage = 100
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> for i in range(7):
        ...     _ = bank.run_one_day(200, 200)
        >>> sum(bank.food_going_bad().get_quantity().values()) > 10 # should be a large value
        True
        """
        # what future stock will look like considering last week's demand
        future_storage = self.next_week_storage_estimate().df
        return Food(future_storage[(future_storage['remaining_days'] <= 7) & (future_storage['quantity'] > 0)])

    @cython.ccall
    def extract_food_from_storage(self, food: Food):
        """Used by simulation to extract food to be sent to other food banks
        :param food: `Food` instace that is the output of this bank's own `food_going_bad`

        >>> initial_storage = 100
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> bank.extract_food_from_storage(Food(10))
        >>> int(sum(bank.get_food_quantity().values())) < initial_storage
        True
        """
        self.storage.subtract(food.get_quantity())

    @cython.ccall
    def receive_food(self, food: Food):
        """
        Receives food
        :param food: Food instance

        >>> initial_storage = 100
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> bank.receive_food(Food(10))
        >>> int(sum(bank.get_food_quantity().values())) < initial_storage
        False
        """
        self.storage.add(food)

    @cython.ccall
    def future_unmet_demand(self) -> Food:
        """
        Estimates future unmet demand. Returns food instances with negative quantities.
        The negative quantities represent what was missing in storage
        :return:

        >>> initial_storage = 10000
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> for i in range(14):
        ...     _ = bank.run_one_day(20_000, 20_000)
        >>> sum(bank.future_unmet_demand().get_quantity().values())
        0
        >>> initial_storage = 1
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> for i in range(14):
        ...     _ = bank.run_one_day(1, 1)
        >>> sum(bank.next_week_storage_estimate().get_quantity().values()) > sum(bank.future_unmet_demand().get_quantity().values())
        True
        """
        future_storage = self.next_week_storage_estimate().df
        return Food(future_storage[future_storage['quantity'] < 0])

    @cython.ccall
    def food_storage(self):
        """API for retreaving food storage dataframe
        :return: storage dataframe

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> -1e-2 < sum(bank.get_food_quantity().values()) - bank.food_storage()['quantity'].sum() < 1e-2
        True
        """
        return self.storage.df.copy()

    @cython.ccall
    def storage_quantities_by_type(self) -> pd.DataFrame:
        """
        Returns storage quantities as a dataframe
        :return:

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> df = bank.storage_quantities_by_type()
        >>> len(df)
        1
        >>> -1e-2 < int(df.sum(axis=1) - 10) < 1e-2
        True
        """
        return self.storage.get_quantity_by_food()

    @cython.ccall
    def run_one_day(self, budget: float, food_donations: float):
        """Runs simulation for the day. Also calls `run_one_day` for each pantry it serves.
        :param budget: Budget for the day
        :param food_donations: Food donations in pounds of food
        :return: Overall waste, demand (based on orders) and utility of all pantries
        """
        self.purchase_food(budget)
        new_food = Food.generate_donation(food_donations)
        self.last_donation = food_donations

        self.storage.add(new_food)

        self.storage.quality_control()

        total_utility = []
        total_waste = None

        day_order_increment = None

        pantries_run = False

        for pantry in self.pantries:
            pantry_output = pantry.run_one_day()
            if pantry_output is None:
                continue  # if pantry wasn't held, we skip
            pantries_run = True
            waste, _, utility, _, true_order = pantry_output
            total_utility.append(utility)
            # total_utility = FoodBank.increment_utility(total_utility, utility)
            total_waste = FoodBank.increment_food_dict(total_waste, waste)
            refined_true_order = self.refine_true_order(true_order)
            day_order_increment = FoodBank.increment_food_dict(day_order_increment, refined_true_order)
            self.update_demand(refined_true_order)

        if not pantries_run:
            return total_waste, self.pantry_demand, 0, None

        self.update_weekly_demand(day_order_increment)

        return total_waste, self.pantry_demand, sum(total_utility), None

    @cython.ccall
    @classmethod
    def refine_true_order(cls, order: Dict[str, float]) -> Dict[str, float]:
        """
        Turns demand food order (based on 3 food types) into real order (based on 5 food types)
        :param order:
        :return:

        >>> FoodBank.refine_true_order({FV: 10, STP: 10, PT: 10}) #doctest: +ELLIPSIS
        {'staples': 10, 'fresh_fruits_and_vegetables': 0, 'packaged_fruits_and_vegetables': 10, 'fresh_protein': 10, 'packaged_protein': 0}
        """
        return {STP: order[STP],
                FFV: (FoodBank.price_demand_choice(order, FFV, PFV, FV))[0],
                PFV: (FoodBank.price_demand_choice(order, FFV, PFV, FV))[1],
                FPT: (FoodBank.price_demand_choice(order, FPT, PPT, PT))[0],
                PPT: (FoodBank.price_demand_choice(order, FPT, PPT, PT))[1]}

    @cython.ccall
    @classmethod
    def price_demand_choice(cls, order: Dict[str, float], food_a: str, food_b: str, food_type: str):
        """
        Converts demand for one of the 3 demand food types into purchase choices based on what is cheapest
        :param order:
        :param food_a:
        :param food_b:
        :param food_type:
        :return:

        >>> Global._base_prices = { STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 }
        >>> order = {FV: 10, STP: 10, PT: 10}
        >>> FoodBank.price_demand_choice(order, FFV, PFV, FV) # PFV is cheaper than FFV, so purchase choice is 0 FFV, 10 PFV
        (0, 10)
        >>> FoodBank.price_demand_choice(order, FPT, PPT, PT)
        (10, 0)
        """
        if Global.price_for(food_a) < Global.price_for(food_b):
            return order[food_type], 0

        return 0, order[food_type]

    @cython.ccall
    def get_food_quantity(self):
        """Returns quantity of food in storage
        :return:

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> -1e-2 < sum(bank.get_food_quantity().values()) - 10 < 1e-2
        True
        """
        return self.storage.get_quantity()

    @cython.ccall
    def get_food_order(self, order):
        """Fulfills given order. Wraper for Food.subtract
        :param order: 
        :return: order result

        >>> initial_storage = 100
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> -1e-2 < sum(bank.get_food_order({ STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 }).get_quantity().values()) - 7 < 1e-2
        True
        >>> bank.get_food_order({ STP: 100, FFV: 2, PFV: 1, FPT: 1, PPT: 2 }) #doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: The "staples" you ordered does not exist or is not sufficient in stock
        """
        return self.storage.subtract(order)

    @cython.ccall
    @classmethod
    def increment_food_dict(cls, total_food_dict, new_food_dict):
        """
        Increments given food dictionary using values of `new_food_dict`
        :param total_food_dict:
        :param new_food_dict:
        :return:

        >>> total = { STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 }
        >>> new = { STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 }
        >>> FoodBank.increment_food_dict(total, new)
        {'staples': 2, 'fresh_fruits_and_vegetables': 4, 'packaged_fruits_and_vegetables': 2, 'fresh_protein': 2, 'packaged_protein': 4}
        """
        if total_food_dict is None:
            return new_food_dict
        return {food: (total_food_dict[food] + food_amount) for food, food_amount in new_food_dict.items()}

    @cython.ccall
    def purchase_food(self, budget: float):
        """Purchases food using given budget
        :param budget: budget in dollars

        >>> initial_storage = 100
        >>> bank = FoodBank(food_insecure_pop=10_000,initial_storage=initial_storage)
        >>> bank.purchase_food(100)
        >>> sum(bank.get_food_quantity().values()) > initial_storage
        True
        """
        demand = self.last_week_demand_proportion()
        types = demand.keys()
        remaining_days = [TYPES[t]['max_days'] for t in types]
        quantity = [demand[t] * budget / Global.price_for(t) for t in types]

        self.last_purchase = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        self.storage.add(self.last_purchase)

    @cython.ccall
    def last_week_total_demand(self) -> float:
        """
        Total demand over all food types
        :return:

        >>> initial_storage = 1000
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> for i in range(14):
        ...     _ = bank.run_one_day(1000, 1000)
        >>> bank.last_week_total_demand() > 0
        True
        """
        return sum([sum(day_order.values()) for day_order in self.last_week_demand])

    @cython.ccall
    def last_week_aggregate_demand(self) -> Dict[str, float]:
        """
        Aggregate demand per food type
        :return:

        >>> initial_storage = 1000
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> for i in range(14):
        ...     _ = bank.run_one_day(1000, 1000)
        >>> sum(bank.last_week_aggregate_demand().values()) > 0
        True
        """
        pantry_demand = {}
        for demand in self.last_week_demand:
            for food, amount in demand.items():
                if food not in pantry_demand:
                    pantry_demand[food] = amount
                    continue
                pantry_demand[food] += amount
        return pantry_demand

    @cython.ccall
    def last_week_demand_proportion(self):
        """Returns demand in proportions. Used to decide what food to buy next.
        Calculation based on last week's demand
        :return: demand proportions

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> for i in range(14):
        ...     _ = bank.run_one_day(1, 1)
        >>> -1e-2 < sum(bank.last_week_demand_proportion().values()) - 1 < 1e-2
        True
        """
        total = self.last_week_total_demand()
        if total == 0:
            return {food: 1 / len(Global.get_food_types()) for food in TYPES.keys()}
        pantry_demand = self.last_week_aggregate_demand()
        return {food: (amount / total) for (food, amount) in pantry_demand.items()}

    @cython.ccall
    def update_demand(self, order):
        """Updates pantry demand values
        :param order: order made by a pantry

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> bank.update_demand({ STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 })
        >>> bank.update_demand({ STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 })
        >>> bank.pantry_demand
        {'staples': 2, 'fresh_fruits_and_vegetables': 4, 'packaged_fruits_and_vegetables': 2, 'fresh_protein': 2, 'packaged_protein': 4}
        """
        for food, amount in order.items():
            self.pantry_demand[food] += amount

    @cython.ccall
    def update_weekly_demand(self, order):
        """
        Updates weekly demand dict
        :param order:
        :return:

        >>> initial_storage = 10
        >>> bank = FoodBank(food_insecure_pop=30_000,initial_storage=initial_storage)
        >>> for i in range(9):
        ...     bank.update_weekly_demand({ STP: 1, FFV: 2, PFV: 1, FPT: 1, PPT: 2 })
        >>> len(bank.last_week_demand) == 7
        True
        >>> bank.last_week_aggregate_demand()
        {'staples': 7, 'fresh_fruits_and_vegetables': 14, 'packaged_fruits_and_vegetables': 7, 'fresh_protein': 7, 'packaged_protein': 14}
        """
        if order is None:
            return
        if len(self.last_week_demand) >= 7:
            self.last_week_demand.pop(0)
        self.last_week_demand.append(order)

    @cython.ccall
    @classmethod
    def increment_utility(cls, total_utility: float, utility: float):
        """Increments total utility
        :param total_utility: 
        :param utility: 
        :return: new total utility

        >>> FoodBank.increment_utility(10, 10)
        20
        """
        if total_utility is None:
            return utility
        else:
            return total_utility + utility


if __name__ == '__main__':
    food_insecure_pop = 323340
    initial_storage = 2267631
    budget = 38614*2
    food_donations = 145376*1.5

    food_bank = FoodBank(food_insecure_pop, initial_storage)
    # Global.add_day()
    Global._base_prices = {
        STP: 2,
        FFV: 3,
        PFV: 2,
        FPT: 3,
        PPT: 2,
    }
    import matplotlib.pyplot as plt
    utility_history = []
    bank_storage = []
    demand_history = []
    quantity_by_food = None
    for day in range(50):
        _, _, utility, _ = food_bank.run_one_day(budget, food_donations)
        bank_storage.append(sum(food_bank.get_food_quantity().values()))
        utility_history.append(utility)
        demand_history.append(food_bank.last_week_total_demand())

        if quantity_by_food is None:
            quantity_by_food = food_bank.storage_quantities_by_type()
        else:
            quantity_by_food = pd.concat([quantity_by_food, food_bank.storage_quantities_by_type()], ignore_index=True)
        
        Global.add_day()

    plt.figure()
    plt.plot(utility_history, label='utility')
    plt.title('Utility')
    plt.show()
    plt.figure()
    plt.plot(bank_storage, label='bank storage')
    plt.title('Bank storage')
    plt.show()
    quantity_by_food.plot()
    plt.show()