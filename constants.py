STP = "staples"
FFV = "fresh_fruits_and_vegetables"
PFV = "packaged_fruits_and_vegetables"
FPT = "fresh_protein"
PPT = "packaged_protein"

FV = "fruits_and_vegetables"
PT = "protein"

# Tentative settings
TYPES = {STP: {"proportion": 0.3, "max_days": 180},
         FFV: {"proportion": 0.1, "max_days": 14},
         PFV: {"proportion": 0.25, "max_days": 360},
         FPT: {"proportion": 0.1, "max_days": 10},
         PPT: {"proportion": 0.25, "max_days": 180}}

PERSON_WEEKLY_DEMAND = {STP: {"mean": 5.125, "std": 0.4},
                        FV: {"mean": 13.25, "std": 1.1},
                        PT: {"mean": 12.875, "std": 0.9}}

# 1.24% households have 7 or more persons. Assume the max number is 10 person, and 7-10 persons are equally likely
FAMILY_DISTRIBUTION = [0.2845, 0.3503, 0.1503, 0.1239, 0.0583, 0.0203] + ([0.0031] * 4)