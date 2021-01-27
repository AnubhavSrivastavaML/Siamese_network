
This repository is to show the working of Siamese Network in tensorflow framework. This repository can be modified to train on other datasets.
Image pair generation is provided in 'mnist_datset_pairs.py' script and can be modified as per use.
This script uses euclidean distance as lamda layer in the network.


Framework : Tensorflow 2.3.1

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


