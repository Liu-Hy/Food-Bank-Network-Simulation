import pandas as pd
from Global import *
from utils import *
from FoodBank import *
from FoodPantry import *

def initialize_food_banks(food_bank_df: pd.DataFrame)->list[FoodBank]:
    """

    :param food_bank_df: dataframe with food bank data per row
    :return: list of food_banks initialized with pre-generated input data
    >>> input_csv=pd.read_csv("input.csv").head(1)
    >>> initialize_food_banks(input_csv)[0]._storage
    blah

    """
    banks=[]
    for row in food_bank_df.itertuples():
        stored_food=row.estimated_pounds_per_year*STOCKPILE_RATIO
        banks.append(FoodBank(row.food_insecure_pop, stored_food))
    return banks

def generate_distance_matrix(food_bank_df: pd.DataFrame)->np.ndarray:
    """

    :param food_bank_df: food_bank_df: dataframe with food bank locations
    :return: array of distances (km) between food bank pairs (foodbank x foodbank)
    """

def generate_funds_distribution(food_bank_df: pd.DataFrame, num_days:int)-> np.ndarray:
    """

    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: numpy array, beta distribution of food purchase funds per day (foodbank x day)
    """

def generate_food_distribution(food_bank_df: pd.DataFrame , num_days:int) -> np.ndarray:
    """
    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: beta distribution of funds per day based on annual budget for food bank
    """
def generate_price_distribution(price_summary: pd.DataFrame , num_days:int) -> dict[np.ndarray]:
    """
    :param num_days: number of days to run simulation
    :param price_path: Dataframe of summary statistics for the good of interest
    :return: dictionary of price distributions for food items and gasoline
    """

def tick_day(food_banks:list):
    """

    :param food_banks:
    :return:
    """
def redistribute_food(food_banks:list):
    """

    :param food_banks: list of food banks to redistribute food between
    :return:
    """
