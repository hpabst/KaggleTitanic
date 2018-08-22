import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from exploratory_analysis import clean_data


# Tutorial taken from https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
def main():
    #Import dataset
    train_data = clean_data(pd.read_csv("train.csv"))

    X_test = pd.read_csv("test.csv")
    ids = X_test["PassengerId"].values
    X_test = clean_data(X_test)
    train_data = train_data.interpolate()
    X_test = X_test.interpolate()
    X_train = train_data
    Y_train = X_train["Survived"]
    X_train.drop(columns=["Survived"], inplace=True)

    gnb = GaussianNB()
    used_features = [
    ]

    #Train classifier
    gnb.fit(
        X_train.values,
        Y_train
    )
    print("Testing predictions.")
    y_test = gnb.predict(X_test)
    with open("results/naive_bayes_test.csv", "w+") as f:
        f.write("PassengerId,Survived\n")
        f.writelines(["{},{}\n".format(ids[i], y_test[i]) for i in range(0, len(y_test))])


    # y_pred = gnb.predict(X_test[used_features])
    #
    # #Print results:
    # print("Number of mislabeled points out of a total {} points: {}, performance {:05.2f}%"
    #       .format(X_test.shape[0],
    #               (X_test["Survived"] != y_pred).sum(),
    #               100 * (1 - X_test["Survived"] != y_pred).sum()/X_test.shape[0]))





if __name__ == "__main__":
    main()