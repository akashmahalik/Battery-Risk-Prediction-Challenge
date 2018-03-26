# Battery-Risk-Prediction-Challenge

# Goal
Given a set of parameters of a battery, we want to predict the risk of the battery. The risk is defined as the number of days before the battery becomes a “bad” battery. See more details in the Data section about the definition of risk.

# Data
We have collected 102,223 records from 588 batteries. Each battery may have more than one record on different dates. Keep in mind there is a date associated with each record, although it has been removed from the data.

 

Each record has 18 attributes.

event_country_code: Country where support contact initiated.

batt_manufacturer: Battery manufacturer. It has been encoded to be anonymous.

installed_count: Number of batteries in the laptop as reported at the date of this record.

batt_instance: Identifies whether this battery is a primary or secondary battery in the laptop.

cycle_count: Number of times that battery has been discharged and recharged.

temperature: Temperature of the battery at the date of this record.

battery_current: Battery electrical current at the date of this record.

design_capacity: Design capacity of the battery.

full_charge_capacity: Full charge capacity of the battery at the date of this record.

remaining_capacity: Remaining battery charge at time of injection date.

design_voltage: Design voltage of the battery.

batt_voltage: Battery voltage at the date of this record.

cell_voltage1: Voltage of battery cell #1 at the date of this record. If the battery contains 2 cells then cell_voltage1 will be 0.  

cell_voltage2: Voltage of battery cell #2 at the date of this record.

cell_voltage3: Voltage of battery cell #3 at the date of this record.   

cell_voltage4: Voltage of battery cell #4 at the date of this record. If the battery contains 2 cells, then cell_voltage4=0 AND cell_voltage1=0.

status_register:  Status register of the battery at the date of this record.

risk: The risk value is defined as the number of days before this battery becomes a “bad” battery. If the battery is “bad” at the date of this record, the risk value is 0. Otherwise, we will find all its later records and look for the first date for the “bad” status. If there’s no “bad” status afterwards, the risk value is -1.


# Evaluation
The evaluation consists of two parts:

1. Can you classify the current status of the battery accurately?

2. Can you forecast the failure of the battery accurately?

 
First, we categorize the risk value into two types: (1) risk = 0 and (2) risk != 0. We can then calculate the F1 score of this binary classification task by treating (1) as the positive label while (2) as the negative label. This F1 is the first score, denoted as F1.

 
Second, we measure the mean relative absolute error for the risk for the records whose groundtruth risks are greater than 0. We denote the predicted risk as P and the groundtruth risk as G for a record. If P is -1, the relative absolute error is defined as 1. Otherwise, the relative absolute error is defined as min(1, |P - G| / G). The mean relative absolute error becomes the second score, denoted as MRAE.


The final score is defined as F1 + (1 - MRAE). All submissions will be ranked by this final score. If there are ties, we will break the tie by the F1 score.
