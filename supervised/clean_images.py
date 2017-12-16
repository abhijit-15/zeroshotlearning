import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

files = os.listdir('./kmeans128/')

test_file = files[0]

c = test_file.split('.')[0]

os.chdir('./kmeans128/')

df = pd.read_csv(test_file)


def plot_hist(df):
	x = df["distance"].as_matrix()
	mu = np.mean(x)
	sigma = np.std(x)
	print(mu , sigma)
	#print(x.shape , x)
	num_bins = 50

	fig, ax = plt.subplots()

	# the histogram of the data
	n, bins, patches = ax.hist(x, num_bins, normed=1)

	# add a 'best fit' line
	y = mlab.normpdf(bins, mu, sigma)
	ax.plot(bins, y, '--')
	ax.set_xlabel('Distances')
	ax.set_ylabel('Probability density')
	ax.set_title(r'Histogram of ' + c)

	fig.tight_layout()
	plt.show()

	return None

df["zscore"] = (df["distance"] - df["distance"].mean())/df["distance"].std(ddof=0) 

print(df.query("zscore > 2")[["path","zscore"]])
