__author__ = 'AndrewSivrit'
import time
print("Loading data")
start_time = time.time()
#from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from scipy import misc
import numpy as np
from PIL import Image
import base64
import re
from io import StringIO
import os
import uuid
import boto
import boto3
from boto.s3.key import Key
from boto.s3.connection import S3Connection

def get_image():
	image_b64 = request.values['imageBase64']
	image_encoded = image_b64.split(',')[1]
	image = base64.decodebytes(image_encoded.encode('utf-8'))
	'digit1-O_n1'.split('n')
	drawn_digit = request.values['digit']
	type = 'O'
	filename = 'digit' + str(drawn_digit) + '-' + type + str(uuid.uuid1()) + '.jpg'
	with open('tmp/' + filename, 'wb') as f:
		f.write(image)

	REGION_HOST = 's3-external-1.amazonaws.com'

	conn = S3Connection(os.environ['AWSAccessKeyId'], os.environ['AWSSecretKey'], host=REGION_HOST)
	bucket = conn.get_bucket('nikitinandrews')

	k = Key(bucket)
	key = filename
	fn = 'tmp/' + filename
	k.key = key
	k.set_contents_from_filename(fn)
	print('Done')
	return filename