import base64
import os
import uuid
import random
import numpy as np

import boto
import boto3
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from codecs import open
from PIL import Image
from scipy.ndimage.interpolation import rotate, shift
from skimage import transform

from two_layer_net import FNN


input_size = 28 * 28
hidden_size = 100
num_classes = 10
net = tln(input_size, hidden_size, num_classes)

stats = net.train(X_train_, y_train_, X_val, y_val,
            num_iters=19200, batch_size=24,
            learning_rate=0.1, learning_rate_decay=0.95,
            reg=0.001, verbose=True)

plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()