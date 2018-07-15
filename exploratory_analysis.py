import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    train_data = pd.read_csv("train.csv")
    encode_sex = {"Sex":{
                            "female":0,
                            "male":1,
    }}
    encode_title = {"Mr.":0,
                    "Miss.":1,
                    "Mrs.":2,
                    "Master.":3}
    if "Title" not in train_data:
        train_data["Title"] = train_data["Name"]
    if "Has_Family" not in train_data:
        train_data["Has_Family"] = train_data["Parch"]
    train_data.replace(encode_sex, inplace=True)
    for i, row in train_data.iterrows():
        title = __get_title(row["Name"])
        train_data.loc[i, "Title"] = encode_title.get(title, 4)
        train_data.loc[i, "Has_Family"] = 1 if (row["Parch"] != 0 or row["SibSp"] != 0) else 0
    corr = train_data.corr()
    return

def __get_title(name):
    return name.split(",")[1].split(" ")[1]


if __name__ == "__main__":
    main()
