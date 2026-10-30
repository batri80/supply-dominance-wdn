"""Convert the BattLeDIM SCADA workbooks to fast pickle files (run once, about 3 minutes).

Reads   <data>/scada/2018_SCADA.xlsx and 2019_SCADA.xlsx
Writes  <outputs>/scada_<year>_<sheet>.pkl for pressures, demands, flows and levels.
"""
import os
import pandas as pd
import config

for year in (2018, 2019):
    src = os.path.join(config.SCADA_DIR, f"{year}_SCADA.xlsx")
    sheets = pd.read_excel(src, sheet_name=None, index_col=0, engine="openpyxl")
    for name, df in sheets.items():
        key = name.split(" ")[0].lower()           # "Pressures (m)" -> "pressures"
        df.to_pickle(f"scada_{year}_{key}.pkl")
        print(f"{year} {key:10s} {df.shape[0]:>7} rows x {df.shape[1]:>3} columns, "
              f"{df.index.min()} to {df.index.max()}, missing {int(df.isna().sum().sum())}")
