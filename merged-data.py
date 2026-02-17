import pandas as pd

# Load files
nov_demand = pd.read_csv("data/extracted/2025/11_November/2025_November_daily_maximum_demand.csv")
dec_demand = pd.read_csv("data/extracted/2025/12_December/2025_December_daily_maximum_demand.csv")

nov_hydro = pd.read_csv("data/extracted/2025/11_November/2025_November_hydro_generation.csv")
dec_hydro = pd.read_csv("data/extracted/2025/12_December/2025_December_hydro_generation.csv")

nov_solar = pd.read_csv("data/extracted/2025/11_November/2025_November_solar_generation.csv")
dec_solar = pd.read_csv("data/extracted/2025/12_December/2025_December_solar_generation.csv")

nov_wind = pd.read_csv("data/extracted/2025/11_November/2025_November_wind_generation.csv")
dec_wind = pd.read_csv("data/extracted/2025/12_December/2025_December_wind_generation.csv")

# Combine months
demand = pd.concat([nov_demand, dec_demand])
hydro = pd.concat([nov_hydro, dec_hydro])
solar = pd.concat([nov_solar, dec_solar])
wind = pd.concat([nov_wind, dec_wind])

# Rename columns
demand = demand.add_prefix("Demand_").rename(columns={"Demand_Date": "Date"})
hydro = hydro.add_prefix("Hydro_").rename(columns={"Hydro_Date": "Date"})
solar = solar.add_prefix("Solar_").rename(columns={"Solar_Date": "Date"})
wind = wind.add_prefix("Wind_").rename(columns={"Wind_Date": "Date"})

# Merge
merged = demand.merge(hydro, on="Date") \
               .merge(solar, on="Date") \
               .merge(wind, on="Date")

merged.to_csv("merged_power_data.csv", index=False)
