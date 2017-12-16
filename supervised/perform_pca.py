import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

path = './4096/'

files = os.listdir(path)

#test_file = files[0]

os.chdir(path)

def perform_PCA(df , n_components = 128):
	'''Takes a dataframe of type "path" , features (4096) , "class" and n_components.
	Returns dataframe of size n_samples * n_components along with path and class '''  	
	
	#For PCA, we need only the numeric columns so we will be ignoring the first 
	#column which is the path to the filename and last column which is the class name
		
	X = df.drop(["path", "class"] , axis = 1)
	
	pca = PCA(n_components)	
	
	#pca_fit = pca.fit(X)
	#print(np.sum(pca_fit.explained_variance_ratio_))
		
	pca_colnames = ["p" + str(i + 1) for i in range(n_components)]
	
	pca_output = pca.fit_transform(X)
	
	pca_df = pd.DataFrame(data = pca_output , columns = pca_colnames , dtype = np.float32)

	path_df , class_df = df["path"].to_frame() , df["class"].to_frame()
	
	res = pd.concat([path_df , class_df , pca_df] , axis = 1)		
		
	return res

out_path = '/home/aphatak/ZSL/pca_analysis/128/'

for f in files:
	try:
		df = pd.read_csv(f)
		res = perform_PCA(df , 128)
		res.to_csv(out_path + f)
		print("PCA for class " + f.split('.')[0] + " completed.")  
	except:
		print("Failed for class " + f.split('.')[0])
		pass
