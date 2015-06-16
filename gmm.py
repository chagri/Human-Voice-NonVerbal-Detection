import numpy as np
from sklearn.mixture import GMM
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold

'''
Build GMM model and outputs average performance on 10 test sets from
10-fold cross validation.
'''
def compute_gmm(features, labels, n_classes):
	#Try GMMs using different types of covariances.
	nb_n_classes = len(n_classes)
	classifiers = dict(
		(covar_type, GMM(n_components=nb_n_classes, covariance_type=covar_type, init_params='wc', n_iter=50))
		for covar_type in ['spherical', 'diag', 'tied', 'full'])
	#Testing K-folds cross validation
	nb_folds = 10
	skf = StratifiedKFold(labels, nb_folds)
	avg_precision = {index: np.zeros([nb_n_classes,]) for index, (name, classifier) in enumerate(classifiers.items())}
	avg_recall = {index: np.zeros([nb_n_classes,]) for index, (name, classifier) in enumerate(classifiers.items())}
	avg_accuracy = {index: 0 for index, (name, classifier) in enumerate(classifiers.items())}
	for train_index, test_index in skf:
		x_train = features[train_index]
		y_train = labels[train_index]
		x_test = features[test_index]
		y_test = labels[test_index]
		n_classes = np.unique(y_train)
		for index, (name, classifier) in enumerate(classifiers.items()):
			# Since we have class labels for the training data, we can
			# initialize the GMM parameters in a supervised manner.
			classifier.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in n_classes])
			# Train the other parameters using the EM algorithm.
			classifier.fit(x_train)
			y_test_pred = classifier.predict(x_test)
			y_test_pred = [int(round(i)) for i in y_test_pred]
			y_test = [int(round(i)) for i in y_test]
			precision = np.around(precision_score(y_test, y_test_pred, average=None),decimals=3)
			accuracy = np.around(accuracy_score(y_test, y_test_pred),decimals=3)
			recall = np.around(recall_score(y_test, y_test_pred, average=None),decimals=3)
			avg_precision[index] = np.add(avg_precision[index],precision)
			avg_recall[index] = np.add(avg_recall[index],recall)
			avg_accuracy[index] = np.add(avg_accuracy[index],accuracy)
	for index, (name, classifier) in enumerate(classifiers.items()):
		avg_precision[index] /= nb_folds
		avg_recall[index] /= nb_folds
		avg_accuracy[index] /= nb_folds
		print "\nclassifier", name
		print "------------------------"
		print "accuracy score", np.around(avg_accuracy[index],decimals=3)
		print "precision score",  np.around(avg_precision[index],decimals=3)
		print 'recall score', np.around(avg_recall[index],decimals=3)
		print "------------------------"