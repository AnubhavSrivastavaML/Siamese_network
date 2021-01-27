from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np




def build(shape):
    input_layer = Input(shape)
    conv1 = Conv2D(8 , (3,3) , padding = 'same' , activation='relu')(input_layer)
    batchnorm1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D((2,2))(batchnorm1)
    
    conv2 = Conv2D(16 , (3,3) , padding = 'same' , activation='relu')(pool1)
    batchnorm2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D((2,2))(batchnorm2)
    
    globalpool = GlobalAveragePooling2D()(pool2)
    
    flatten_layer = Flatten()(globalpool)
    
    dense = Dense(32,activation='relu')(flatten_layer)
    
    model = Model(inputs = input_layer , outputs = dense)
    
    return model
    

