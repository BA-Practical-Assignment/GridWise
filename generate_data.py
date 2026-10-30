#Generating SYNTHETIC data used when the real datasets are not found.
import json
import random

import numpy as np
import pandas as pd
import config

# 1. Synthetic hourly electricity demand (MW)
def generate_demand(years: int = 4, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=years * 365 * 24, freq="h")
    hour = idx.hour.values
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values

    base = 15000.0
    trend = np.linspace(0, 600, len(idx))                                   
    daily = 2600 * np.sin((hour - 9) / 24 * 2 * np.pi) \
        + 900 * np.sin((hour - 3) / 12 * 2 * np.pi)                         
    weekly = np.where(dow >= 5, -1400, 0)                                  
    yearly = 2200 * np.cos((doy - 20) / 365 * 2 * np.pi) \
        + 1200 * np.cos((doy - 200) / 365 * 4 * np.pi)                      
    noise = rng.normal(0, 450, len(idx))
    demand = base + trend + daily + weekly + yearly + noise

    # public holidays: demand drops
    holiday_mask = np.array([(d.month, d.day) in {(1, 1), (7, 4), (12, 25), (12, 26)} for d in idx])
    demand[holiday_mask] -= 1800

    # a few random heat-wave episodes (demand spikes)
    for _ in range(years * 2):
        start = rng.integers(0, len(idx) - 96)
        demand[start:start + 72] += rng.uniform(1500, 3000)

    return pd.DataFrame({"Datetime": idx, "DEMAND_MW": demand.round(1)})

# 2. Synthetic customer fault reports
LOCATIONS = ["Main Road", "the school", "Church Street", "our street", "the clinic", "Block C",
             "the shopping centre", "Ward 7", "the taxi rank", "Park Avenue", "the industrial area",
             "Section B", "Mandela Drive", "the hospital", "our house","Collegepark", "the stadium","the university", "the police station", "the fire station",]

TEMPLATES = {
    "outage": [
        "There is no electricity at {loc} since this morning",
        "Power went off at {loc} and has not come back",
        "We have been without power for hours near {loc}",
        "the lights are out in {loc} please help",
        "Total blackout at {loc} please send a team",
        "electricity is off again at {loc}",
        "No power in the whole area around {loc}",
        "our power tripped and the whole of {loc} is dark",
    ],
    "billing": [
        "My electricity bill is too high this month",
        "I was charged twice for my account",
        "I paid my bill but it still shows outstanding",
        "The amount on my statement does not match my meter reading",
        "I want to query my electricity account balance",
        "I never received my bill for last month",
        "Why was I billed for estimated usage",
        "my prepaid token was bought but the units did not load",
    ],
    "meter_fault": [
        "My meter is not displaying anything",
        "The prepaid meter keeps beeping and shows an error",
        "Meter at {loc} is running too fast",
        "The electricity meter box is damaged and sparking",
        "my smart meter stopped working",
        "Meter reading looks wrong and the display is flickering",
        "The meter will not accept my token",
        "Our meter was tampered with and needs inspection",
    ],
    "streetlight": [
        "The street light at {loc} is not working",
        "Streetlights have been off for a week at {loc}",
        "A street light is flickering all night near {loc}",
        "the lamp post outside {loc} is dark and it is unsafe",
        "Several street lights are on during the day at {loc}",
        "No lights on the road at {loc} please fix",
    ],
    "cable_damage": [
        "A power line has fallen down at {loc}",
        "There are exposed wires hanging near {loc}",
        "A cable is sparking on the ground at {loc}",
        "The transformer at {loc} is smoking and making a loud noise",
        "A tree fell on the electricity cables at {loc}",
        "Broken pole and live wires on the road at {loc}",
        "A cable was cut by construction workers at {loc}",
        "pole is leaning and the wires are very low at {loc}",
    ],
}

FILLERS = ["", "", "", " please urgent", " thanks", " this is the second time I am reporting",
           " children are at home", " hello", " can someone assist", " ASAP"]


def _typo(text: str, rng: random.Random, p: float = 0.04) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _urgency(category: str, text: str) -> str:
    t = text.lower()
    if category == "cable_damage" or any(w in t for w in ["sparking", "smoking", "fallen", "live wires", "unsafe"]):
        return "high"
    if category in {"outage", "meter_fault"} or "hospital" in t or "clinic" in t:
        return "medium"
    return "low"


def generate_fault_reports(n_per_class: int = 300, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for cat, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            text = rng.choice(templates).format(loc=rng.choice(LOCATIONS)) + rng.choice(FILLERS)
            if rng.random() < 0.5:
                text = _typo(text, rng)
            if rng.random() < 0.3:
                text = text.lower()
            rows.append({"report": text.strip(), "category": cat, "urgency": _urgency(cat, text)})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
