### In this file, we write some helper functions to help CRF.py.
import pandas as pd 

def peek():
    # peek the data 
    df = pd.read_parquet("./data/alien_train.parquet")
    print(df.head())
    print(df.info())
    # print(df.columns)
    # columns = df.columns
    # print(type(columns))
    # for col in columns:
    #     print(col)

if __name__ == "__main__":
    peek()