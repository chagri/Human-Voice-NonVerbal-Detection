import numpy as np
from nolearn.dbn import DBN
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold

'''
Build Deep belief network model and outputs average performance on 10 test sets from
10-fold cross validation.
'''
def compute_dbn(features, labels, n_classes):
	nb_folds = 10
	nb_n_classes = len(n_classes)
	skf = StratifiedKFold(labels, nb_folds)
	avg_precision = np.zeros([nb_n_classes,])
	avg_recall = np.zeros([nb_n_classes,])
	avg_accuracy = 0
	for train_index, test_index in skf:
		x_train = features[train_index]
		y_train = labels[train_index]
		x_test = features[test_index]
		y_test = labels[test_index]
		clf_DBN = DBN([x_train.shape[1], 300, nb_n_classes], learn_rates=0.3, learn_rate_decays=0.9, epochs=10,verbose=0)
		clf_DBN.fit(x_train, y_train)
		y_test_pred = clf_DBN.predict(x_test)
		y_test_pred = [int(i) for i in y_test_pred]
		precision = precision_score(y_test, y_test_pred, average=None)
		recall = recall_score(y_test, y_test_pred,average=None)
		avg_accuracy += accuracy_score(y_test, y_test_pred)
		avg_precision = np.add(avg_precision,precision)
		avg_recall = np.add(avg_recall,recall)
	avg_precision /= nb_folds
	avg_recall /= nb_folds
	avg_accuracy /= nb_folds
	print "-------------Testing DBN Accuracy------------"
	print "accuracy score", np.around(avg_accuracy,decimals=3)
	print "precision score",  np.around(avg_precision,decimals=3)
	print 'recall score', np.around(avg_recall,decimals=3)