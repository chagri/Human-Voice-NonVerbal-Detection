import os
import wave
import contextlib
from math import ceil

'''
.txt files originally only list the different interesting class (cough, laugh...) but not the
speech. Here we are reintegrating the speech and defining it by the no_label_class parameter.
Note that time unit is by default 0.1sec. We use the audio to define the end of file.
'''
def expand_labels(path_to_audio, path_to_txt, no_label_class, extension_txt=".txt"):
	path_to_txt_out = path_to_txt[:-len(extension_txt)] + "_labels.txt"
	if not os.path.isfile(path_to_txt_out):
		print "expanding labels of file",path_to_txt_out.split("/")[-1]
		with contextlib.closing(wave.open(path_to_audio, 'r')) as f_audio:
			frames = f_audio.getnframes()
			rate = f_audio.getframerate()
			duration = int(ceil(float(frames) / rate * 10))
		f = open(path_to_txt, 'r')
		start = 0
		str_to_write = ""
		for line in f:
			line = line.rstrip().split(',')
			line = [int(i) if i else None for i in line]
			flag = line[2]
			if flag == 0:
				temp = line[0] - start
				for i in range(start, temp + start):
					str_to_write += str(i) + "," + str(no_label_class) + '\n'
				start = line[0]
			elif flag > 0:
				temp = line[0] - start
				for i in range(start, temp + start):
					str_to_write += str(i) + ',' + str(line[1]) + '\n'
				start = line[0]
			else:
				if duration < start:
					return None
				for i in range(start, duration):
					str_to_write += str(i) + ',' + str(no_label_class) + '\n'
				break
		f_out = open(path_to_txt_out, "w")
		f_out.write(str_to_write)
	return path_to_txt_out
