import numpy as np
import tensorflow as tf
import re
import csv
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

LOGDIR = "logs"
MODELDIR = "models"

def read_test_data():
    return __read_csv_file('test.csv')

def read_train_data():
    return __read_csv_file('train.csv')

def __read_csv_file(filename):
    with open(filename) as csvfile:
        rows = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows

def encode_passenger(passenger):
    embarks = {"C": 0., "Q": 1., "S": 2., "":3.}
    title_map = {"Mr":1., "Miss":2., "Mrs":3., "Master":4., "Rare":5.}
    regex = re.search(' ([A-Za-z]+)\.', passenger.get("Name"))
    title = regex.group(1)
    if title not in title_map.keys():
        title = "Rare"
    temp_array = []
    #temp_array.append(float(passenger["PassengerId"]))
    temp_array.append(float(passenger["Pclass"]))
    if passenger["Sex"] == "male":
        temp_array.append(0.)
    else:
        temp_array.append(1.)
    #temp_array.append(float(passenger["SibSp"]))
    #temp_array.append(float(passenger["Parch"]))
    #temp_array.append(float(passenger["Fare"]))
    temp_array.append(embarks[passenger["Embarked"]])
    if passenger.get("Age") == "":
        temp_array.append(0.)
    else:
        temp_array.append(float(passenger.get("Age")))
    temp_array.append(title_map.get(title))
    return np.array(temp_array)

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def forward_prop(X, parameters):
    """
    Forward propagation for a n-layer neural network using RELU function.
    :param X: Input dataset placeholder of shape input_size x num_examples
    :param parameters: Parameters for each layer W1, b1, W2, b2, etc.
    :return: Computation graph for output of output layer.
    """
    num_layers = int(len(parameters)/2)
    Zcurr = tf.add(tf.matmul(parameters["W1"], X), parameters["b1"])
    Acurr = tf.nn.relu(Zcurr)
    for i in range(2,num_layers+1):
        Wcurr = parameters["W"+str(i)]
        bcurr = parameters["b"+str(i)]
        Zcurr = tf.add(tf.matmul(Wcurr, Acurr), bcurr)
        tf.summary.histogram("pre_activations", Zcurr)
        Acurr = tf.nn.relu(Zcurr)
        tf.summary.histogram("activations", Acurr)
    return Acurr

def init_parameters(num_units, seed=True):
    """
    Initializes parameters for a neural network with given number of layers and shape for each layer
    :param num_hidden_units: Array of number of hidden units for each layer. num_units[0] corresponds to initial input size.
    :return:
    """
    if seed:
        tf.set_random_seed(1)
    parameters = {}
    for i in range(1, len(num_units)):
        with tf.name_scope("fc"+str(i)):
            parameters["W"+str(i)] = tf.get_variable("Weight_"+str(i), [num_units[i], num_units[i-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters["b"+str(i)] = tf.get_variable("bias_"+str(i), [num_units[i], 1], initializer=tf.zeros_initializer())
            #variable_summaries(parameters["W"+str(i)])
            #variable_summaries(parameters["b"+str(i)])
    return parameters

def compute_cost(AL, Y):
    """
    Compute cost function for model.
    :param AL: Output of last layer in model.
    :param Y: Placeholder variable for "true" labels.
    :return: Tensor for cost computation.
    """
    with tf.name_scope("cost"):
        cost = tf.losses.mean_squared_error(labels=Y, predictions=AL)
    tf.summary.scalar("cost", cost)
    return cost

def compute_cost_with_regularizarion(AL, Y, reg_lambda = 0.01):
    """

    :param AL:
    :param Y:
    :param W:
    :return:
    """
    lossL2 = tf.add_n([tf.nn.l2_loss(i) for i in tf.trainable_variables() if 'bias' not in i.name])
    return tf.add(tf.losses.mean_squared_error(labels=Y, predictions=AL), reg_lambda*lossL2)

def shuffle_in_unison(a, b):
    """
    Shuffles 2 arrays in unison. i.e. Both before and after shuffling a[i] and b[i] correspond to each other.
    Sourced from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison.
    :param a:
    :param b:
    :return:
    """
    np.random.seed(5)
    rng_state = np.random.get_state()
    np.random.shuffle(a.T)
    np.random.set_state(rng_state)
    np.random.shuffle(b.T)

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs=2000, minibatch_size=32,
          print_cost=True, seed=True, layers=None, show_plot = False, regularize = False, threshold=0.9):
    """
    Runs a neural network based on provided parameters.
    :param X_train: Training data.
    :param Y_train: Labels for training data.
    :param X_test: Testing data.
    :param Y_test: Labels for testing data.
    :param learning_rate: Learning rate for optimizer.
    :param num_epochs: Number of training iterations to perform.
    :param minibatch_size: Size of minibatches to use.
    :param print_cost: Print current cost after every 100 epochs?
    :param seed: Seed number generator?
    :param layers: List of number of units per hidden layer (e.g. for a 3 layer network with 8 units in first layer, 4 in second, and 2 in third: [8, 4, 2]). Default is [100, 50, output_layer_size].
    :return: (trained parameters, layers, training set accuracy, testing set accuracy)
    """
    ops.reset_default_graph()
    writer = tf.summary.FileWriter(LOGDIR)
    if(seed):
        tf.set_random_seed(1)
    (input_size, num_examples) = X_train.shape
    costs = []
    train_accuracies = []
    test_accuracies = []
    X_place = tf.placeholder(tf.float32, [input_size, None], name="X")
    Y_place = tf.placeholder(tf.float32, [Y_train.shape[0], None], name="Y")
    if layers is None:
        layer_list = [input_size, 100, 50, Y_train.shape[0]]
    else:
        layer_list = [input_size] + layers
    params = init_parameters(layer_list, seed=True)
    AL = forward_prop(X_place, params)
    if regularize:
        cost_func = compute_cost_with_regularizarion(AL, Y_place)
    else:
        cost_func = compute_cost(AL, Y_place)
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_func)
    init = tf.global_variables_initializer()
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.greater(AL, threshold), tf.equal(Y_place, 1.0))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar("test_accuracy", accuracy)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer.add_graph(sess.graph)
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(num_examples/minibatch_size)
            X_minibatches = np.array_split(X_train, num_minibatches, axis=1)
            Y_minibatches = np.array_split(Y_train, num_minibatches, axis=1)
            for i in range(num_minibatches):
                minibatchX = X_minibatches[i]
                minibatchY = Y_minibatches[i]
                _, minibatch_cost = sess.run([optimizer, cost_func], feed_dict={X_place:minibatchX, Y_place:minibatchY})
                epoch_cost += minibatch_cost / int(num_examples/minibatch_size)
            train_accuracy = accuracy.eval({X_place: X_train, Y_place: Y_train})
            test_accuracy = accuracy.eval({X_place: X_test, Y_place: Y_test})
            summary = sess.run(merged, feed_dict={X_place: X_test, Y_place: Y_test})
            writer.add_summary(summary, epoch)
            if print_cost and epoch%100 == 0:
                print("Cost after epoch %i: %f" %(epoch, epoch_cost))
                print("Train Accuracy:", train_accuracy)
                print("Test Accuracy:", test_accuracy)
                test_accuracies.append(test_accuracy)
                train_accuracies.append(train_accuracy)
            if print_cost and epoch % 2 == 0:
                costs.append(epoch_cost)
        trained_params = sess.run(params)
        #Plot cost
        if show_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations per 10s')
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()
            plt.plot(np.squeeze(train_accuracies))
            plt.ylabel("Accuracy")
            plt.xlabel("Iterations per 100")
            plt.title("Training Accuracies")
            plt.show()
            plt.plot(np.squeeze(test_accuracies))
            plt.ylabel("Accuracy")
            plt.xlabel("Iterations per 100")
            plt.title("Test Accuracies")
            plt.show()
        final_test_accuracy = accuracy.eval({X_place: X_test, Y_place: Y_test})
        final_train_accuracy = accuracy.eval({X_place: X_train, Y_place: Y_train})
        results = tf.round(AL.eval({X_place:X_train}))
        tp = tf.count_nonzero(results * Y_train).eval()
        tn = tf.count_nonzero((results-1) * (Y_train-1)).eval()
        fp = tf.count_nonzero((results * (Y_train-1))).eval()
        fn = tf.count_nonzero((results-1)*Y_train).eval()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1score = (2*precision*recall)/(precision+recall)
        print("Model training set results:")
        print("Accuracy:            {:10.2f}%".format((final_train_accuracy)*100))
        print("True Positives:      {:10.2f}%".format((tp/X_train.shape[1])*100))
        print("True Negatives:      {:10.2f}%".format((tn/X_train.shape[1])*100))
        print("False Positives:     {:10.2f}%".format((fp/X_train.shape[1])*100))
        print("False Negatives:     {:10.2f}%".format((fn/X_train.shape[1])*100))
        print("Precision:           {:10.2f}".format(precision))
        print("Recall:              {:10.2f}".format(recall))
        print("F1 Score:            {:10.2f}".format(f1score))
        print()
        results = tf.round(AL.eval({X_place: X_test}))
        tp = tf.count_nonzero(results * Y_test).eval()
        tn = tf.count_nonzero((results - 1) * (Y_test - 1)).eval()
        fp = tf.count_nonzero((results * (Y_test - 1))).eval()
        fn = tf.count_nonzero((results - 1) * Y_test).eval()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = (2 * precision * recall) / (precision + recall)
        num_members = X_test.shape[1]
        print("Model test set results:")
        print("Accuracy:            {:10.2f}%".format((final_test_accuracy) * 100))
        print("True Positives:      {:10.2f}%".format((tp / num_members) * 100))
        print("True Negatives:      {:10.2f}%".format((tn / num_members) * 100))
        print("False Positives:     {:10.2f}%".format((fp / num_members) * 100))
        print("False Negatives:     {:10.2f}%".format((fn / num_members) * 100))
        print("Precision:           {:10.2f}".format(precision))
        print("Recall:              {:10.2f}".format(recall))
        print("F1 Score:            {:10.2f}".format(f1score))
        print()
    return (trained_params, layer_list, train_accuracy, test_accuracy)



def main():
    training_data = read_train_data()
    X = np.array([encode_passenger(i) for i in training_data]).T
    Y = np.array([float(i.get("Survived", -1.)) for i in training_data])
    num_examples = X.shape[1]
    Y = Y.reshape((num_examples, 1)).T
    np.set_printoptions(precision=3)
    print("X shape:" + str(X.shape))
    print("Y shape:" + str(Y.shape))
    breakpoint = int(0.8*X.shape[1])
    shuffle_in_unison(X, Y)
    X_train, X_dev = np.hsplit(X, [breakpoint])
    Y_train, Y_dev = np.hsplit(Y, [breakpoint])
    print("X_train shape:" + str(X_train.shape))
    print("X_dev shape  :" + str(X_dev.shape))
    print("Y_train shape:" + str(Y_train.shape))
    print("Y_dev shape  :" + str(Y_dev.shape))
    result = model(X_train, Y_train, X_dev, Y_dev, layers=[100, 100, 50, 1],
                   show_plot=True, learning_rate=0.001, num_epochs=2000, threshold=0.5)
    #Issue with judging accuracy. Currently only going by what percentage of labels are 1.
    return


if __name__ == "__main__":
    main()