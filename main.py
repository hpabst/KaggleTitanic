from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
from keras.utils.vis_utils import plot_model
from datetime import datetime
from exploratory_analysis import clean_data
from Metrics import Metrics
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pickle
import keras.backend as K


THRESHOLD = 0.65

def get_input_matrices(data, shuffle=True):
    if shuffle:
        data = data.sample(frac=1)
    y = data["Survived"].values
    x = data.drop(columns=["Survived"]).values
    print("X shape: {0}".format(x.shape))
    print("Y shape: {0}".format(y.shape))
    return x, y


def create_model(x, normalize=True, layers=None):
    if layers is None:
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
    else:
        model = Sequential()
        model.add(Dense(layers[0], input_dim=x.shape[1], activation='relu'))
        for i in range(1, len(layers)):
            model.add(Dense(layers[i], activation='relu'))
            if normalize:
                model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, epochs=20000, batch_size=10, evaluate=True):
    metrics = Metrics()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=[x_test, y_test],
                        verbose=2)
    scores = model.evaluate(x_train, y_train)
    if evaluate:
        print("\n{0}: {1}".format(model.metrics_names[1], scores[1]))
    return model, history, scores


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


def predict_data(filepath, model, evaluate=False):
    assert(os.path.isfile(filepath))
    test_data = pd.read_csv(filepath)
    ids = test_data["PassengerId"].values
    test_data = clean_data(test_data)
    if "Survived" in test_data:
        test_data.drop(columns=["Survived"], inplace=True)
    predictions = model.predict(test_data.values)
    #rounded = [int(round(x[0])) for x in predictions]
    up_indices = predictions >= THRESHOLD
    down_indices = predictions < THRESHOLD
    predictions[up_indices] = 1
    predictions[down_indices] = 0
    if evaluate:
        print("PassengerId,Survived")
        for i in range(0, len(predictions)):
            print("{0}, {1}".format(ids[i], predictions[i]))
    return ids, predictions


def save_model_and_results(model, history, test_res, train_res, train_score, filename):
    path = "results/{0}".format(filename)
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(path+"/train_results.csv", "w+") as f:
        train_ids, train_predictions = train_res
        f.write("{0}: {1}%\n".format(model.metrics_names[1], train_score[1]*100))
        f.write("PassengerId,Survived\n")
        f.writelines(["{0},{1}\n".format(train_ids[i], train_predictions[i]) for i in range(0, len(train_ids))])
    with open(path+"/test_results.csv", "w+") as f:
        test_ids, test_preds = test_res
        f.write("PassengerId,Survived\n")
        f.writelines(["{0},{1}\n".format(test_ids[i], test_preds[i]) for i in range(0, len(test_ids))])
    plot_model(model, to_file=path+"/model.png", show_shapes=True)
    model.save(path+"/model.h5")
    with open(path+"/history", "wb") as f:
        pickle.dump(history.history, f)
    return


def main():
    x_train, y_train = read_training_data(filepath="train.csv")
    models = [[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 8],
              #[32, 16, 8, 4, 2],
              ]
    for m in models:
        print(m)
        model = create_model(x_train, layers=m)
        model, history, scores = train_model(model, x_train, y_train, epochs=500)
        test_ids, test_predictions = predict_data("test.csv", model)
        train_ids, train_predictions = predict_data("train.csv", model)
        current_dt_str = datetime.now().strftime("%B-%d-%Y-%I%M%p")
        save_model_and_results(model,
                               history,
                               (test_ids, test_predictions),
                               (train_ids, train_predictions),
                               scores,
                               current_dt_str)
    return


if __name__ == "__main__":
    main()
