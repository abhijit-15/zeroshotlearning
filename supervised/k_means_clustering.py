import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

files = os.listdir('./128/')

#test_file = files[0]

os.chdir('./128/')

def dist_from_centroid(df):
	'''Computes Euclidean distance between p1...p128 and c1...c128
	for each sample in the dataframe and retur	ns the series'''			
	d = []	
	for j in range(len(df)):
		vec_dist = [df.iloc[j]["p" + str(i + 1)] - df.iloc[j]["c" + str(i + 1)] for i in range(128)]	
		d.append(np.linalg.norm(vec_dist))	
	dist = np.array(d)	
	return dist

def perform_kmeans(df , n = 8 , max_iter = 1000):
	'''Used to perform k-means on the 128 dimensional representation
	of the image encodings after PCA (4096 -> 128). Takes dataframe as
	"path","class", p1 --> p128. Adds assigned cluster centroids and cluster labels'''
	usecols = ["p" + str(i + 1) for i in range(128)]	
	
	X = df[usecols]	

	kmeans = KMeans(n , random_state = 1).fit(X)	
	
	labels = kmeans.labels_ #Index of centroid assigned
	labels_df = pd.DataFrame(labels , columns = ["Label"])	

	centroids = kmeans.cluster_centers_ #Centroid coordinates in 128 dimensional space 	
	
	assigned_centers = np.array(centroids[labels]) #Centroid for each point
	ccols = ["c" + str(i + 1) for i in range(128)]	
	
	centers = pd.DataFrame(data = assigned_centers , columns = ccols , dtype = np.float32)
	
	res = pd.concat([df , labels_df , centers] , axis = 1)	
	
	dist = dist_from_centroid(res)
	dist_df = pd.DataFrame(data = dist , columns = ["distance"] , dtype = np.float32)
 	
	result = pd.concat([res , dist_df] , axis = 1)
	
	return result	

out_path = '/home/aphatak/ZSL/k_means/kmeans128/'

for f in files:
	try:
		df = pd.read_csv(f)
		result = perform_kmeans(df , n = 8 , max_iter = 1000)
		result.to_csv(out_path + f)
		print("KMeans for class " + f.split('.')[0] + " completed.")  
	except:
		print("Failed for class " + f.split('.')[0])
		pass
