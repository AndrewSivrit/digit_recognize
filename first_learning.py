import base64
import os
import uuid
import random
import numpy as np
import matplotlib.pyplot  as plt

from codecs import open
from PIL import Image
from scipy.ndimage.interpolation import rotate, shift
from skimage import transform

from two_layer_net_first_learning import FNN

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


input_size = 28 * 28
hidden_size = 100
num_classes = 10
net = FNN(input_size, hidden_size, num_classes)

X, y = augment('train')
X_val, y_val = augment('val')



stats = net.train(X, y, X_val, y_val,
            num_iters=19200, batch_size=24,
            learning_rate=0.1, learning_rate_decay=0.95,
            reg=0.001, verbose=True)

np.save('tmp/weights.npy', net.params)   

plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss function')
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.show()


plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()