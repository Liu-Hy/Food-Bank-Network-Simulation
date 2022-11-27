STP = "staples"
FFV = "fresh_fruits_and_vegetables"
PFV = "packaged_fruits_and_vegetables"
FPT = "fresh_protein"
PPT = "packaged_protein"

FV = "fruits_and_vegetables"
PT = "protein"

# tentative settings
TYPES = {STP: {"proportion": 0.3, "max_days": 180},
         FFV: {"proportion": 0.1, "max_days": 14},
         PFV: {"proportion": 0.25, "max_days": 360},
         FPT: {"proportion": 0.1, "max_days": 10},
         PPT: {"proportion": 0.25, "max_days": 180}}

PERSON_WEEKLY_DEMAND = {STP: {"mean": 5.125, "std": 0.4},
                        FV: {"mean": 13.25, "std": 1.1},
                        PT: {"mean": 12.875, "std": 0.9}}
