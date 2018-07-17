from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
from exploratory_analysis import clean_data
import os
import pandas as pd


def get_input_matrices(data, shuffle=True):
    if shuffle:
        data = data.sample(frac=1)
    y = data["Survived"].values
    x = data.drop(columns=["Survived"]).values
    print("X shape: {0}".format(x.shape))
    print("Y shape: {0}".format(y.shape))
    return x, y


def create_model(x):
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, epochs=20000, batch_size=10, evaluate=True):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    if evaluate:
        scores = model.evaluate(x_train, y_train)
        print("\n{0}: {1}".format(model.metrics_names[1], scores[1]))
        model.save("models/{0}.h5".format(scores[1]))
    return model


def read_training_data(filepath="train.csv"):
    assert(os.path.isfile(filepath))
    train_data = pd.read_csv(filepath)
    train_data = clean_data(train_data)
    x, y = get_input_matrices(train_data)
    return x, y


def read_testing_data(filepath="test.csv"):
    assert(os.path.isfile(filepath))
    test_data = pd.read_csv(filepath)
    test_data = clean_data(test_data)
    x = test_data.values
    print("X Test Shape: {0}".format(x.shape))
    return x


def predict_test_data(filepath, model, model_filename="default"):
    assert(os.path.isfile(filepath))
    test_data = pd.read_csv(filepath)
    ids = test_data["PassengerId"].values
    test_data = clean_data(test_data)
    predictions = model.predict(test_data.values)
    rounded = [int(round(x[0])) for x in predictions]
    with open("models/{0}.csv".format(model_filename), "w+") as f:
        f.write("PassengerId,Survived\n")
        f.writelines(["{0},{1}\n".format(ids[i], rounded[i]) for i in range(0, len(rounded))])
    return


def main():
    x_train, y_train = read_training_data(filepath="train.csv")
    model = create_model(x_train)
    model = train_model(model, x_train, y_train, epochs=2000)
    model = load_model("models/0.8451178459874976.h5")
    input("Press anything to continue.")
    predict_test_data("test.csv", model ,model_filename="0.8451178459874976.h5")
    return


if __name__ == "__main__":
    main()