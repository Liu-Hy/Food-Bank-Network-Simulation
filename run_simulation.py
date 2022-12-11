import pandas as pd

import Global
from Global import *
from utils import *
from FoodBank import *
from FoodPantry import *
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
import cython

geod = Geodesic.WGS84


@cython.cfunc
def initialize_food_banks(food_bank_df: pd.DataFrame) -> list[FoodBank]:
    """
    Use a combination of previous group data (food insecure population and geocoding) and manually annotated data
    (pounds of food distributed and programming budget) to initialize food banks
    :param food_bank_df: dataframe with food bank data per row
    :return: list of food_banks initialized with pre-generated input data
    >>> input_csv=pd.read_csv("input.csv").head(1)
    >>> len(initialize_food_banks(input_csv)[0].pantries) #(323340/5)/500 = 64 rounded down
    64


    """

    banks = []
    for row in food_bank_df.itertuples():
        stored_food = row.estimated_pounds_per_year * STOCKPILE_RATIO
        banks.append(FoodBank(row.food_insecure_pop, stored_food))
    return banks


@cython.cfunc
def generate_distance_matrix(food_bank_df: pd.DataFrame) -> np.ndarray:
    """
    generate pairwise distance between food banks
    :param food_bank_df: food_bank_df: dataframe with food bank locations
    :return: array of distances (km) between food bank pairs (foodbank x foodbank)

    >>> input_csv=pd.read_csv("input.csv").head(2)
    >>> generate_distance_matrix(input_csv)[0,1]# doctest: +ELLIPSIS
    269.74...

    """
    lat = food_bank_df["latitude"]
    long = food_bank_df["longitude"]
    num_bank = len(lat)
    distance = np.empty([num_bank, num_bank])

    for i in range(0, num_bank):
        for j in range(0, num_bank):
            pair_dist = geod.Inverse(lat[i], long[i], lat[j], long[j])['s12']
            pair_dist = pair_dist / 1609.34  # convert meters to miles
            distance[i, j] = pair_dist

    return distance


@cython.cfunc
def generate_funds_distribution(food_bank_df: pd.DataFrame, num_days: int) -> np.ndarray:
    """
    generates randomized distribution for daily food purchase budget
    daily budget distributed randomly but approach annual budget
    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: numpy array, beta distribution of food purchase funds per day (foodbank x days)

    >>> input_csv=pd.read_csv("input.csv").head(5)
    >>> test=generate_funds_distribution(input_csv, 365)[0]
    >>> 76000<np.mean(test)<78000
    True
    >>> 14000000<np.sum(test)<142000000
    True
    """
    budgets = food_bank_df["estimated_budget"]
    ret = np.zeros(shape=(len(budgets), num_days))
    for i in range(0, len(budgets)):
        annual = budgets[i]
        total = annual * FOOD_PURCHASE_BUDGET_RATIO  # annual budget for food purchases
        daily_avg = total / 365
        min_val = daily_avg * 0.5  # min is half daily avg
        max_val = daily_avg * 2  # max is twice daily avg
        std_dev = daily_avg * 0.2  # estimate standard deviation as 20%
        ret[i] = mod_beta_random(low=min_val, high=max_val, mean=daily_avg, std=std_dev, samples=num_days)
    return ret


@cython.cfunc
def generate_food_distribution(food_bank_df: pd.DataFrame, num_days: int) -> np.ndarray:
    """
    Generates distrbution of daily food donations based on observed data,
    Based on Beta distribution
    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: beta distribution of food donations per day (foodbank x day)
    >>> input_csv=pd.read_csv("input.csv").head(5)
    >>> test=generate_funds_distribution(input_csv, 365)[0]
    >>> 76000<np.mean(test)<78000
    True
    >>> 27000000<np.sum(test)<29000000
    True
    >>> np.max(test)<130000
    True

    """
    amounts = food_bank_df["estimated_pounds_per_year"]
    ret = np.zeros(shape=(len(amounts), num_days))
    for i in range(0, len(amounts)):
        total = amounts[i]
        daily_avg = total / 365
        min_val = daily_avg * 0.5  # min is half daily avg
        max_val = daily_avg * 2  # max is twice daily avg
        std_dev = daily_avg * 0.3  # estimate standard deviation
        ret[i] = mod_beta_random(low=min_val, high=max_val, mean=daily_avg, std=std_dev, samples=num_days)
    return ret


@cython.cfunc
def generate_good_prices(price_summary: pd.DataFrame, num_days: int, ) -> dict[list[np.ndarray]]:
    """
    Use statistics from Bureau of Labor Statistics to generate price distributions for each good
    :param num_days: number of days to run simulation
    :param price_path: Dataframe of summary statistics for the good of interest
    :return: dictionary of price distributions for food items and gasoline
    >>> input_df=pd.read_csv("price_summary.csv")
    >>> output=generate_good_prices(input_df, 365)
    >>> 2.4<np.mean(output[GAS])<3.5
    True

    """
    price_dict = dict()

    price_dict[GAS] = good_price_distr(price_summary, "gas", num_days)
    price_dict[FFV] = good_price_distr(price_summary, "ffv", num_days)
    price_dict[FPT] = good_price_distr(price_summary, "meat", num_days)
    price_dict[STP] = good_price_distr(price_summary, "staples", num_days)
    price_dict[PFV] = good_price_distr(price_summary, "ffv", num_days) * PACKAGED_COST_RATIO  # modify packaged price
    price_dict[PPT] = good_price_distr(price_summary, "meat", num_days) * PACKAGED_COST_RATIO  # modify packaged price

    return price_dict


@cython.cfunc
def good_price_distr(price_summary: pd.DataFrame, good: str, num_days: int,
                     random_seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Generate price distribution
    starts at current price and performs a random walk
    resistance grows as price diverges from the mean



    :param random_seed: random seed
    :param price_summary: df of price data
    :param good: string name of row
    :param num_days: number of days
    :return: distribution of good price
    >>> prices=pd.read_csv("price_summary.csv")
    >>> gas=good_price_distr(prices, "gas", 365)
    >>> 2.4<np.mean(gas)<3.5
    True
    >>> np.std(gas)<0.3
    True
    """
    price_summary = price_summary.set_index("good")
    mean = price_summary.loc[good]["mean"]
    real_price = price_summary.loc[good]["latest_price"]

    mean_delta = price_summary.loc[good]["mean_delta"]
    mean_delta = mean_delta / 30  # monthly mean change to daily mean change
    std_delta = price_summary.loc[good]["std_delta"]
    std_delta = std_delta / math.sqrt(30)  # monthly std to daily std
    price_list = [mean]  # start random walk from historical mean

    for i in range(1, num_days):
        prev = price_list[i - 1]  # start from previous day price
        scale_factor = (mean - prev) / mean  # trend toward mean food price
        scaled_change = mean_delta * scale_factor
        change = np.random.normal(scaled_change, std_delta)  # normal distribution centered around price change
        price_list.append(prev + change)  # calculate new price
    return np.array(price_list)


@cython.cfunc
def precalculate_supply_demand(food_banks: list[FoodBank]) -> tuple[list[Food], list[Food]]:
    """
    pre-calls and stores results of food_going_bad, quality control, and future_unmet_demand
    :param food_banks:
    :return: tuple of list of Food objects (sending foodbank results, receiving foodbank results)
    >>> fb=[FoodBank(10000,3000, ), FoodBank(10000,3000, )]
    >>> sup_demand=precalculate_supply_demand(fb)
    >>> sup_demand[0][0].get_quantity()[FFV]!=0
    True
    >>> sup_demand[1][0].get_quantity()[FFV]!=0
    False


    """
    # precalculate food bank demands and supplies
    excess_supply = []
    excess_demand = []
    for f in food_banks:
        # precalculate supply and remove going bad food
        supply = f.food_going_bad()
        supply.quality_control()
        excess_supply.append(supply)

        # precalculate demands for each food bank
        excess_demand.append(f.future_unmet_demand())
    return (excess_supply, excess_demand)


@cython.cfunc
def generate_net_demand(num_foodbank: int, excess_supply_demand: tuple[list[Food]]) -> np.ndarray:
    """
    calculates net demand per food type by taking the min and max of each food type demanded

    :param num_foodbank: number of food banks
    :param excess_supply_demand: excess supply and demand Food objects per food bank
    :return: 3d numpy array of foodtype, sending foodbank, receiving food bank

    >>> fb=[FoodBank(100000,300000, ), FoodBank(100000,300000, )]
    >>> sup_demand=precalculate_supply_demand(fb)
    >>> np.shape(generate_net_demand(2,sup_demand))
    (5, 2, 2)
    """

    net_food_demand = np.zeros(
        [len(FOOD_GOODS), num_foodbank, num_foodbank])  # food type by food bank by food bank matrix
    for i in range(num_foodbank):
        available_food = excess_supply_demand[0][i].get_quantity()
        for j in range(num_foodbank):
            demanded_food = excess_supply_demand[1][j].get_quantity()
            for k in range(len(FOOD_GOODS)):
                food = FOOD_GOODS[k]
                net_food_demand[k, i, j] = min(available_food[food], -demanded_food[food])
    return net_food_demand

@cython.cfunc
def food_network(food_banks: list, distance_mat: np.ndarray, payment_source: str = "recipient", ) -> list[float]:
    """
    Implements national food sharing network
    food banks indexed by (sending,receiving)

    :param payment_source: setting to control distribution strategy,
    recipient: receiving food bank pays
    sender: sending food bank pays
    disaster: government (no transportation cost)
    :param distance_mat: distance matrix between pairs of food banks
    :param food_banks: list of food banks to redistribute food between
    :return: Returns monetary cost of transport paid per food bank

    Difficult to write doctests, as there are many dependencies in the Food Bank class and the markov chain nature
    of the project

    """
    num_foodbank = len(food_banks)
    food_market_value = np.empty([num_foodbank, num_foodbank])

    excess_supply_demand = precalculate_supply_demand(food_banks)
    net_demand = generate_net_demand(num_foodbank, excess_supply_demand)

    pounds_to_move = np.sum(net_demand, axis=0)
    pounds_to_move[pounds_to_move < 1000] = 0  # set threshold for reasonable transport
    for i in range(len(FOOD_GOODS)):
        food_market_value += net_demand[i] * Global.price_for(FOOD_GOODS[i])

    cost_per_truck = distance_mat / TRUCK_MPG * Global._gas_price * 2  # return trip for truck
    cost_per_lb_truck = cost_per_truck / POUNDS_PER_TRUCK

    total_cost = cost_per_lb_truck * pounds_to_move

    cost_effective = total_cost - food_market_value  # gap between transport cost and market price represents economic value

    cost_effective[cost_effective < 0] = 0
    top_efficiency = np.argmax(cost_effective,
                               axis=0)  # max of each column is most efficient truck for given food bank to send

    costs = [0] * num_foodbank
    # take top non-zero value of each row, and exchange the food between the food banks

    for i in range(0, len(top_efficiency)):
        if cost_effective[i, top_efficiency[i]] != 0:
            j = top_efficiency[i]
            if payment_source == "recipient":
                costs[j] = total_cost[i, j]
            if payment_source == "sender":
                costs[i] = total_cost[i, j]
            for k in range(len(FOOD_GOODS)):
                amount = net_demand[k, i, j]  # access amount of food to transfer for each food type
                food_exchange(food_banks[i], food_banks[j], excess_supply_demand[0][i], FOOD_GOODS[k], amount)

    return costs


@cython.cfunc
def food_exchange(sender: FoodBank, recipient: FoodBank, food:Food, type:str,amount:float):

    """

    send specific food type if sufficient
    :param sender: Foodbank sending food
    :param recipient: Foodbank receiving food
    :param food: amount of food to exchange
    :param type: food category to transfer
    :param amount: amount of food to transfer
    :return: None
    >>> FB1=FoodBank(10000,20000)
    >>> FB2 = FoodBank (10000,1)
    >>> excess=precalculate_supply_demand([FB1,FB2])
    >>> food_exchange(FB1, FB2, excess[0][0], STP, 100)
    >>> FB2.get_food_quantity()[STP]>1
    True
    """

    sub_food=food.select(type)
    if sub_food.get_quantity()[type]>=amount:
        try:
            sender.extract_food_from_storage(sub_food)
            recipient.receive_food(sub_food)
        except ValueError:
            pass



@cython.cfunc
def run_one_bank(arg_tuple: Tuple) -> tuple:
    """
    wrapper function for multiprocessing
    :param tuple(food_bank, daily budget, daily donations, index i, index j, transportation costs)

    :return: tuple of tick one day output
    """

    total_waste, pantry_demand, total_utility, _ = arg_tuple[0].run_one_day(arg_tuple[1][arg_tuple[4], arg_tuple[3]]-
                                                                            arg_tuple[5][4],arg_tuple[2][arg_tuple[4],
                                                                            arg_tuple[3]])
    return total_waste, pantry_demand, total_utility


if __name__ == "__main__":


    ###CONFIG VARIABLES (SETTABLE)###
    num_days = 28  # number of days to run simulation
    inflation_rate = 1.08
    num_food_banks=10
    network_distribution=False

    if network_distribution:
        network_desc="WITH_EXCHANGE"
    else:
        network_desc = "WITHOUT_EXCHANGE"

    pool = Pool()  # initialize MP pool

    food_banks_df = pd.read_csv("input.csv").head(num_food_banks)  # number of food banks to initialized
    prices_df = pd.read_csv("price_summary.csv")
    # initialize Global state
    global_state = Global()
    global_state._price_inflation_pct = inflation_rate

    # initialize food banks
    food_banks = initialize_food_banks(food_banks_df)
    distances = generate_distance_matrix(food_banks_df)

    # generate randomized distributions
    daily_budget = generate_funds_distribution(food_banks_df, num_days)
    daily_donations = generate_food_distribution(food_banks_df, num_days) * DONATION_BOOST
    good_prices = generate_good_prices(prices_df, num_days)

    #plot price distributions
    days = np.arange(0, num_days)
    plt.plot(days, good_prices[FFV])
    plt.title("Fresh fruit and vegetable price per day")
    plt.savefig("plots/FFV_price_distr.png")
    plt.ylabel('Price ($)')
    plt.xlabel('Day')
    plt.show()

    #plot donation distribution
    plt.plot(days, daily_donations[0])
    plt.title("Donations ($) per day")
    plt.savefig("plots/donations.png")
    plt.ylabel('Donations ($)')
    plt.xlabel('Day')
    plt.show()

    """
    plt.plot(days, daily_budget[0])
    plt.show()
    """
    #initialize empty dicts
    daily_waste = dict()
    for d in FOOD_GOODS:
        daily_waste[d] = [0] * num_days
    weekly_utility = []
    weekly_total = 0

    for i in range(0, num_days):
        if i % 7 == 0:
            weekly_utility.append(weekly_total)
            weekly_total = 0

        print("Running day " + str(i))
        for g in good_prices:
            if g == GAS:
                Global._gas_price = good_prices[g][i] * Global._price_inflation_pct
            else:
                Global.set_price(g, good_prices[g][i])

        Global.base_prices()  # apply inflation
        transportation_costs=[0]*len(food_banks)
        if i > 0 and network_distribution:
            transportation_costs=food_network(food_banks, distances)
        args = []  # args for multiprocess
        for j in range(0, len(food_banks)):
            curr = food_banks[j]
            args.append((curr, daily_budget, daily_donations, i, j, transportation_costs))

        output = pool.map(run_one_bank, args)  # run all food bank day method calls in parallel
        for r in output:
            if r[0]:
                for w in r[0]:
                    daily_waste[w][i] += r[0][w]
            # print(pantry_demand)
            if r[2]:
                weekly_total += r[2]

        Global.add_day()

    weeks = np.arange(0, len(weekly_utility))

    plt.plot(weeks, weekly_utility[0:])
    plt.title("Total utility per week")
    plt.ylabel('Total utility')
    plt.xlabel('Week number')
    plt.savefig("plots/sim_level_utility_" + network_desc)
    plt.show()
    print("Average weekly utility:")
    print(np.mean(weekly_utility))
    print("Min weekly utility:")
    print(np.min(weekly_utility))

    print()


    print("Average waste:")
    for w in daily_waste:
        print(w)
        print(np.mean(daily_waste[w]))
        plt.plot(days, daily_waste[w], label=w)
    plt.title("Daily waste by type")
    plt.ylabel('Wasted food')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig("plots/sim_level_waste_"+network_desc)

    plt.show()



    pool.close()
