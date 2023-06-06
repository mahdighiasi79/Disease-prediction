import helper_functions as hf
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("Dataset3.csv")
    feature = df["disease"]
    print(hf.HasMissingValue(feature))
    print(hf.ExtractValues(feature))
