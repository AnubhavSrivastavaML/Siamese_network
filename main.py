from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from mnist_dataset_pairs import make_pairs
from custom_layers import euclidean_distance
import tensorflow as tf
from plot import plot_training
from model import build
import numpy as np


#=========================  Dataset  Creation from tensorflow default dataset for convenience but any datset can be used  ===============================
dataset = mnist.load_data()

(train_images,train_labels),(test_images,test_labels)=dataset

train_images = np.expand_dims(train_images,axis=-1)
test_images = np.expand_dims(test_images,axis=-1)
(train_pair_images , train_pair_labels) = make_pairs(train_images,train_labels)
(test_pair_images , test_pair_labels) = make_pairs(test_images,test_labels)


#   ============================    model building section [model architecture can be changed in model.py script] ============================================
 
model = build((28,28,1))
tf.keras.utils.plot_model(model)
print(model.summary())
'''
Model Architecture	:::		Model: "functional_21"
					_________________________________________________________________
					Layer (type)                 Output Shape              Param #   
					=================================================================
					input_19 (InputLayer)        [(None, 28, 28, 1)]       0         
					_________________________________________________________________
					conv2d_34 (Conv2D)           (None, 28, 28, 8)         80        
					_________________________________________________________________
					batch_normalization_4 (Batch (None, 28, 28, 8)         32        
					_________________________________________________________________
					max_pooling2d_4 (MaxPooling2 (None, 14, 14, 8)         0         
					_________________________________________________________________
					conv2d_35 (Conv2D)           (None, 14, 14, 16)        1168      
					_________________________________________________________________
					batch_normalization_5 (Batch (None, 14, 14, 16)        64        
					_________________________________________________________________
					max_pooling2d_5 (MaxPooling2 (None, 7, 7, 16)          0         
					_________________________________________________________________
					global_average_pooling2d_9 ( (None, 16)                0         
					_________________________________________________________________
					flatten_9 (Flatten)          (None, 16)                0         
					_________________________________________________________________
					dense_10 (Dense)             (None, 32)                544       
					=================================================================
					Total params: 1,888
					Trainable params: 1,840
					Non-trainable params: 48
'''


img1= Input(shape = (28,28,1))
img2= Input(shape = (28,28,1))
featureExtractor = build((28,28,1))
feat1 = featureExtractor(img1)
feat2 = featureExtractor(img2)


distance = Lambda(euclidean_distance)([feat1,feat2])    #  lamda function to calculate euclidean distance but can be other measures in custom layers file    
outputs = Dense(1,activation='sigmoid')(distance)
model_final = Model(inputs=[img1,img2],outputs=outputs)


print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
# train the model
print("[INFO] training model...")
history = model.fit(
    [train_pair_images[:, 0], train_pair_images[:, 1]], train_pair_labels[:],
    validation_data=([test_pair_images[:, 0], test_pair_images[:, 1]], test_pair_labels[:]),
    batch_size=16, 
    epochs=10)
    
    
plot_training(history,"")

