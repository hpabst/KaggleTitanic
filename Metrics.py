import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

#Code for this class taken from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class Metrics(Callback):
    def __init__(self):
        super().__init__()
        self.val_f1s = []
        self.val_recalls = []
        self.precisions = []

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.precisions = []
        return

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.precisions.append(_val_precision)
        print("- val_f1: {0} - val_precision: {1} - val_recall {2}".format(_val_f1, _val_precision, _val_recall))
        return