"""
Name: Craig Brooks
PHSX 815 Spring 2023
HW # 12
Due Date 4/24/2023
This code performs K-means clustering using the K-means algorithm, Gaussian Mixture Models, and GMM+K-means
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM

if __name__ == "__main__":


	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s [-c -n]" % sys.argv[0])
		print
		sys.exit(1)
	if '-c' in sys.argv:
		p = sys.argv.index('-c')
		clusters = int(sys.argv[p+1])
	else:
		clusters = 3
	if '-n' in sys.argv:
		p = sys.argv.index('-n')
		samples = int(sys.argv[p+1])
	else:
		samples = 100

	if '-std' in sys.argv:
		p = sys.argv.index('-n')
		stdev = int(sys.argv[p+1])
	else:
		stdev = 1
	np.random.seed(677)

	# generates random data based on number of samples, clusters, and standard deviation
	X, y_true = make_blobs(n_samples=samples, centers=clusters,
                       cluster_std=stdev, random_state=42)
	#X = X[:, ::-1] # flip axes for betteplotting
	
	# Initialization of KMeans, GMM and GMM w/ Kmeans
	kmeans = KMeans(clusters, random_state=42)
	gmm = GMM(clusters, random_state=42, tol=1e-4, init_params = 'random' )
	gk = GMM(clusters, random_state=42, tol=1e-4, init_params = 'kmeans' )

	# Fitting of the models 
	k_fit = kmeans.fit_predict(X)
	g_fit = gmm.fit_predict(X)
	gk_fit = gk.fit_predict(X)
	
	#plots the data
	plt.scatter(X[:, 0], X[:, 1], c=k_fit, s=15, cmap='viridis')
	plt.title(f'KMeans (True), {clusters} clusters, {samples} samples')
	#plt.savefig(f'kmeans_{clusters}_{samples}_{stdev}')
	plt.show()
	plt.scatter(X[:, 0], X[:, 1], c=g_fit, s=15, cmap='viridis')
	plt.title(f'Gaussian Mixture Model, {clusters} clusters, {samples} samples')
	#plt.savefig(f'gaussianmm_{clusters}_{samples}_{stdev}')
	plt.show()
	plt.scatter(X[:, 0], X[:, 1], c=gk_fit, s=15, cmap='viridis')
	plt.title(f'Gaussian Mixture Model using KMeans, {clusters} clusters, {samples} samples')
	#plt.savefig(f'gmm_kmeans_{clusters}_{samples}_{stdev}')
	plt.show()
