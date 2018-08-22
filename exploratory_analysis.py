import pandas as pd
import numpy as np
import math


def main():
    train_data = pd.read_csv("train.csv")
    clean_data(train_data)
    return


def clean_data(df):
    encode_sex = {
        "Sex": {
                            "female": 0,
                            "male": 1,
        }
    }
    encode_title = {"Mr.": 0,
                    "Miss.": 1,
                    "Mrs.": 2,
                    "Master.": 3}
    if "Title" not in df:
        df["Title"] = np.vectorize(__get_title)(df["Name"])
        df["Title"] = np.vectorize(encode_title.get)(df["Title"], 4)
    if "Has_Family" not in df:
        df["Has_Family"] = np.vectorize(__encode_has_family)(df["Parch"], df["SibSp"])
    if "Is_Child" not in df:
        df["Is_Child"] = np.vectorize(__encode_is_child)(df["Age"])
    if "Num_Cabins" not in df:
        df["Num_Cabins"] = np.vectorize(__encode_num_cabins)(df["Cabin"])
    df.replace(encode_sex, inplace=True)
    df.drop(columns=["Age", "SibSp", "Parch", "Name", "Ticket", "Cabin", "Embarked", "PassengerId"], inplace=True)
    new_df = pd.DataFrame()
    for col in df:
        if col != "Fare" and col != "Survived":
            one_hot = pd.get_dummies(df[col])
            one_hot = one_hot.add_prefix(col)
            for new_name in list(one_hot):
                new_df[new_name] = one_hot[new_name]
    fare_max = df["Fare"].max()
    new_df["Fare"] = (df["Fare"]/fare_max)
    if "Survived" in df:
        new_df["Survived"] = df["Survived"]
    return new_df


def __encode_num_cabins(cabin):
    if type(cabin) is float and math.isnan(cabin):
        return 0
    else:
        cabins = cabin.split(" ")
        return len(cabins)


def __get_title(name):
    return name.split(",")[1].split(" ")[1]


def __encode_is_child(age):
    return 1 if age < 16 else 0


def __encode_has_family(parch, sibsp):
    return 1 if (parch + sibsp > 0) else 0


if __name__ == "__main__":
    main()
