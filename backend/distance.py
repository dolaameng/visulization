import numpy as np 
from scipy.spatial import distance
import sys

def read_voc(fname):
	with open(fname, 'r') as f:
		lines = f.readlines()[1:]
		nwords = len(lines)
		nfeats = len(lines[0].split()) - 1
		vocab = [None] * nwords
		features = np.empty((nwords, nfeats), dtype=np.float)
		for i, line in enumerate(lines):
			line = line.split()
			vocab[i] = line[0]
			features[i, :] = np.asarray(line[1:], dtype=np.float)
		return vocab, features 

def build_dist_matrix(features):
	dm = 1 - distance.squareform(distance.pdist(features, 'cosine'))
	return dm 

def find_knn(dm, voc, word, k):
	try:
		iword = voc.index(word)
	except ValueError, ex:
		return None
	knn_index = np.argsort(dm[iword])[-k:][::-1]
	return [(voc[i], dm[iword][i]) for i in knn_index]

if __name__ == '__main__':
	k = 40
	voc_fname = sys.argv[1]
	vocab, features = read_voc(voc_fname)
	dm = build_dist_matrix(features)
	while True:
		word = raw_input('Enter word or sentence (EXIT to break):')
		if word == 'EXIT':
			break
		else:
			print find_knn(dm, vocab, word, 40)
