from flask import Flask, render_template, request, redirect
import requests

import sys
sys.path.insert(0, '/Users/shreyashpandey/acads/cs229/project/')

from test_resnet18 import *

import urllib

#Using Standardd Static and Template Folders. So no need to redefine.
app = Flask(__name__)

def getTags(url):
	urllib.urlretrieve(url, "/Users/shreyashpandey/acads/cs229/project/images/test.jpg")
	tags = get_hashtags("/Users/shreyashpandey/acads/cs229/project/images/test.jpg")
	return tags

@app.route('/',methods=['GET' , 'POST'])
def main(tags=None):
	print("executing")
	if request.method == 'POST':
		url = request.form.get('url')
		tags = getTags(url)
		print(tags)
		return render_template('index.html', tags=tags , img_disp=url)
	return render_template('index.html',tags='', img_disp='')


if __name__ == "__main__":
    app.run(debug=True)