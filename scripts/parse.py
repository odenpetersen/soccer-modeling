import pandas as pd

data = "../data/figshare/"
competitions = ["England","Germany","World_Cup","France","Spain","Italy","European_Championship"]

def df_from_json(filename):
    with open(filename,"r") as f:
        df = pd.read_json(f)
    return df

def parse():
    df = pd.concat([df_from_json(data + f"events_{competition}.json") for competition in competitions])
    return df

if __name__ == "__main__":
    print(parse())
