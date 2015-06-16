from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold
import numpy as np

'''
Build HMM model and outputs average performance on 10 test sets from
10-fold cross validation.
'''
def compute_hmm(features, labels, n_classes):
	num_classes = len(n_classes)
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
		model = GaussianHMM(num_classes, covariance_type="diag", n_iter=1000)
		model.fit([x_train])
		y_train_pred = model.predict(x_train)
		correlation_mat = np.zeros([num_classes, num_classes])
		matches = [-1 for i in xrange(num_classes)]
		for real_dex in xrange(num_classes):
			t_count = 0
			for dex in xrange(len(y_train)):
				if y_train[dex] == real_dex:
					t_count += 1
					pred = y_train_pred[dex]
					correlation_mat[real_dex, pred] += 1
			if not t_count == 0:
				for dex in xrange(num_classes):
					correlation_mat[real_dex, dex] /= t_count
		while correlation_mat.max() > 0:
			mv = correlation_mat.max()
			[row_, col_] = np.where(correlation_mat==mv)
			correlation_mat[row_, :] = np.zeros([1, num_classes])
			correlation_mat[:, col_] = np.zeros([num_classes, 1])
			matches[col_] = row_
		if min(matches) < 0:
			print 'Error Correlating Stuff'
			break
		y_test_pred = model.predict(x_test)
		y_test_pred = [matches[int(i)] for i in y_test_pred]
		precision = precision_score(y_test, y_test_pred, average=None)
		recall = recall_score(y_test, y_test_pred,average=None)
		avg_accuracy += accuracy_score(y_test, y_test_pred)
		avg_precision = np.add(avg_precision,precision)
		avg_recall = np.add(avg_recall,recall)
	avg_precision /= nb_folds
	avg_recall /= nb_folds
	avg_accuracy /= nb_folds
	print "-------------Testing HMM Accuracy------------"
	print "accuracy score", np.around(avg_accuracy,decimals=3)
	print "precision score",  np.around(avg_precision,decimals=3)
	print 'recall score', np.around(avg_recall,decimals=3)