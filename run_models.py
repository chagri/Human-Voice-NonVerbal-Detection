import os.path
import numpy as np
from format import expand_labels
from get_features import get_data
from dbn import compute_dbn
from hmm import compute_hmm
from gmm import compute_gmm

global extension_audio
global extension_txt_file

'''Extract features and associated class of all audio and combine them'''
def combine_data(list_path_to_audio, list_path_to_txt, nb_features=13):
	combine_n_classes = set()
	first_time = 1
	for index in range(len(list_path_to_audio)):
		path_to_audio = list_path_to_audio[index]
		path_to_txt = list_path_to_txt[index]
		#st = time.time()
		txt_file = expand_labels(path_to_audio, path_to_txt, 5)
		if not txt_file:
			print "Error: in expand_labels function of txt file", path_to_txt
		else:
			#print "expand_labels function, done in...", time.time() - st
			print txt_file.split("/")[-1], ": getting features..."
			[data_features, labels, n_classes] = \
				get_data(path_to_audio, txt_file, ",", nb_features)
			if not first_time:
				combine_data_features = np.concatenate((combine_data_features, data_features))
				combine_labels = np.concatenate((combine_labels, labels))
			else:
				combine_data_features = data_features
				combine_labels = labels
				first_time = 0
			combine_n_classes.update(n_classes)
			print "DONE"
	return [combine_data_features, combine_labels, combine_n_classes]

'''Get all audios .wav and associated .txt files in audio directory'''
def ParseDirectory(full_path):
	list_path_to_audio = []
	list_path_to_txt = []
	dirList = os.listdir(full_path)
	for item_name in dirList:
		item_path = os.path.join(full_path, item_name)

		if(os.path.isdir(item_path)):
			all_lists = ParseDirectory(item_path)
			list_path_to_audio.extend(all_lists[0])
			list_path_to_txt.extend(all_lists[1])
		elif ".wav" in item_name:
			txt_file = os.path.join(full_path, item_name.replace(".wav", ".txt"))
			if (os.path.isfile(txt_file)):
				list_path_to_txt.append(txt_file)
				list_path_to_audio.append(item_path)
	return [list_path_to_audio, list_path_to_txt]


extension_txt_file = ".txt"
extension_audio = ".wav"
file_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
all_lists = ParseDirectory(os.path.join(file_path, 'audio'))
list_path_to_audio = all_lists[0]
list_path_to_txt = all_lists[1]

#number of features considered
nb_features = 13
[data_features, target, n_classes] = combine_data(list_path_to_audio, list_path_to_txt,nb_features )

if list_path_to_txt:
	'''Here you should add the models you want to test and compute results for'''
	#GMM
	#compute_gmm(data_features, target, n_classes)
	#DBN
	compute_dbn(data_features, target, n_classes)
	#HMM
	#compute_hmm(data_features, target, n_classes)
	print "This is a test on DBN model. Please refer to run_models.py file to test another model"