import tensorflow as tf
import numpy as np
import os
import random
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot  as plt

from PIL import Image

from skimage import transform

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_test_images = mnist.test.images.reshape(-1, 28, 28, 1)
batch_size = 128

# =============================================================================
# def augment( image, label):
# =============================================================================
def augment( folder):
    
    
    ims_add = []
    labs_add = []
        
    files = os.listdir(path="./tmp/" + folder) 
    random.shuffle(files)
    for f in files:
        image = Image.open('./tmp/' + folder + '/' + f)
#        print(f)
       # f.write(image)
        

        angles = np.arange(-30, 30, 5)
        bbox = Image.eval(image, lambda px: 255-px).getbbox()
        widthlen = bbox[2] - bbox[0]
        heightlen = bbox[3] - bbox[1]
        if heightlen > widthlen:
            widthlen = int(20.0 * widthlen/heightlen)
            heightlen = 20
        else:
            heightlen = int(20.0 * widthlen/heightlen)
            widthlen = 20
            
        hstart = int((28 - heightlen) / 2)
        wstart = int((28 - widthlen) / 2)
        for i in [min(widthlen, heightlen), max(widthlen, heightlen)]:
            for j in [min(widthlen, heightlen), max(widthlen, heightlen)]:
                resized_img = image.crop(bbox).resize((i, j), Image.NEAREST)
                resized_image = Image.new('L', (28,28), 255)
                resized_image.paste(resized_img, (wstart, hstart))
                
                angles_ = random.sample(set(angles), 6)
                for angle in angles_:
                    transformed_image = transform.rotate(np.array(resized_image), angle, cval=255, preserve_range=True).astype(np.uint8)
                    
#                    images = Image.fromarray(transformed_image.reshape(28,28), 'L')
                    
#                    images.save('tmp/result/' + f)
                    
                    labs_add.append(int(f[5]))
                    img_temp = Image.fromarray(np.uint8(transformed_image))
                    imgdata = list(img_temp.getdata())
                    
                    normalized_img = [(255.0 - x) / 255.0 for x in imgdata]
                    
                    ims_add.append(normalized_img)
        
    image_array = np.array(ims_add)
    label_array = np.array(labs_add)
        
    return image_array, label_array

X_train, y_train = augment('train')
X_val, y_val = augment('val')

trX = X_train.reshape(-1, 28, 28, 1) # 28x28x1
teX = X_val.reshape(-1, 28, 28, 1)
enc = OneHotEncoder()
enc.fit(y_val.reshape(-1, 1), 10) #.toarray() # 10x1
trY = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
teY = enc.fit_transform(y_val.reshape(-1, 1)).toarray()

# Define architecture
def model(X, w, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(
                     tf.nn.conv2d(
                     X, w,
                     strides=[1, 1, 1, 1],
                     padding='SAME'
                     )
                     + b1
                     )
    l1 = tf.nn.max_pool(
                        l1a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
    )
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3a = tf.nn.relu(
                     tf.nn.conv2d(
                     l1, w3,
                     strides=[1, 1, 1, 1],
                     padding='SAME'
                     )
                     + b3
                     )

    l3 = tf.nn.max_pool(
                        l3a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME'
    )
    # Reshaping for dense layer
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) + b5
    return pyx

tf.reset_default_graph()

# Define variables
init_op = tf.global_variables_initializer()

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = tf.get_variable("w", shape=[4, 4, 1, 16],
                            initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="b1", shape=[16],
                            initializer=tf.zeros_initializer())
w3 = tf.get_variable("w3", shape=[4, 4, 16, 32],
                            initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="b3", shape=[32],
                            initializer=tf.zeros_initializer())
w4 = tf.get_variable("w4", shape=[32 * 7 * 7, 625],
                            initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="b4", shape=[625],
                            initializer=tf.zeros_initializer())
w_o = tf.get_variable("w_o", shape=[625, 10],
                            initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable(name="b5", shape=[10],
                            initializer=tf.zeros_initializer())
# Dropout rate
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w, w3, w4, w_o, p_keep_conv, p_keep_hidden)

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y) + reg_constant * sum(reg_losses))

train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#Training
train_acc = []
val_acc = []
test_acc = []
train_loss = []
val_loss = []
test_loss = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # Training iterations
    for i in range(256):
        # Mini-batch
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end],
                                          Y: trY[start:end],
                                          p_keep_conv: 0.8,
                                          p_keep_hidden: 0.5})
        # Comparing labels with predicted values
        train_acc2 = np.mean(np.argmax(trY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: trX,
                                                         Y: trY,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        train_acc.append(train_acc2)
        
        val_acc2 = np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         Y: teY,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        val_acc.append(val_acc2)
        test_acc2 = np.mean(np.argmax(mnist.test.labels, axis=1) ==
                         sess.run(predict_op, feed_dict={X: mnist_test_images,
                                                         Y: mnist.test.labels,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        test_acc.append(test_acc2)
        print('Step {0}. Train accuracy: {3}. Validation accuracy: {1}. \
Test accuracy: {2}.'.format(i, train_acc2, val_acc2, test_acc2))
        
        _, loss_train = sess.run([predict_op, cost],
                              feed_dict={X: trX,
                                         Y: trY,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        train_loss.append(loss_train)
        _, loss_val = sess.run([predict_op, cost],
                               feed_dict={X: teX,
                                         Y: teY,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        val_loss.append(loss_val)
        _, loss_test = sess.run([predict_op, cost],
                              feed_dict={X: mnist_test_images,
                                         Y: mnist.test.labels,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})
        test_loss.append(loss_test)
        print('Train loss: {0}. Validation loss: {1}. \
Test loss: {2}.'.format(loss_train, loss_val, loss_test))
    # Saving model
    all_saver = tf.train.Saver() 
    all_saver.save(sess, '/resources/data.chkp')
    
plt.subplot(2, 1, 1)
plt.plot('train_loss')
plt.title('Loss function train')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()

plt.subplot(2, 1, 1)
plt.plot('val_loss')
plt.title('Loss function val')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()

plt.subplot(2, 1, 1)
plt.plot('test_loss')
plt.title('Loss function test')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()

plt.subplot(2, 1, 2)
plt.plot('train_acc', label='train')
plt.plot('val_acc', label='val')
plt.plot('test_acc', label='test')
plt.title('Classification accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Predicting
with tf.Session() as sess:
    # Restoring model
    saver = tf.train.Saver()
    saver.restore(sess, "/resources/data.chkp")

    # Prediction
    pr = sess.run(predict_op, feed_dict={X: mnist_test_images,
                                         Y: mnist.test.labels,
                                         p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0})

    print(np.mean(np.argmax(mnist.test.labels, axis=1) ==
                         sess.run(predict_op, feed_dict={X: mnist_test_images,
                                                         Y: mnist.test.labels,
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))