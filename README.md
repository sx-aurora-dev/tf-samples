# TensorFlow-VE Samples

    % pip install Pillow
    % wget https://upload.wikimedia.org/wikipedia/commons/b/bc/Elephant.jpg

## ResNet50 inference

```
# VE (Type 10B)
% python keras_applications/predict.py --batch 32 -m resnet50 -n 5 --nchw Elephant.jpg --disable-eager
Elapsed 0.930 sec for 32 images. 34.404 images/sec
Elapsed 0.188 sec for 32 images. 170.026 images/sec
Elapsed 0.189 sec for 32 images. 169.759 images/sec
Elapsed 0.189 sec for 32 images. 169.552 images/sec
Elapsed 0.189 sec for 32 images. 169.482 images/sec

# CPU (Xeon Gold 6126 x 2)
% VE_NODE_NUMBER=-1 python keras_applications/predict.py --batch 48 -m resnet50 -n 5 Elephant.jpg --disable-eager
Elapsed 1.037 sec for 48 images. 46.302 images/sec
Elapsed 0.394 sec for 48 images. 121.967 images/sec
Elapsed 0.399 sec for 48 images. 120.375 images/sec
Elapsed 0.406 sec for 48 images. 118.340 images/sec
Elapsed 0.428 sec for 48 images. 112.141 images/sec
Elapsed 0.414 sec for 48 images. 115.974 images/sec
```

## MobileNetV2 inference

```
# VE (Type 10B)
% python keras_applications/predict.py --batch 32 -m mobilenet_v2 -n 5 --nchw Elephant.jpg --disable-eager
Elapsed 0.938 sec for 32 images. 34.120 images/sec
Elapsed 0.093 sec for 32 images. 342.538 images/sec
Elapsed 0.083 sec for 32 images. 383.757 images/sec
Elapsed 0.083 sec for 32 images. 384.349 images/sec
Elapsed 0.083 sec for 32 images. 385.324 images/sec

# CPU (Xeon Gold 6126 x 2)
% VE_NODE_NUMBER=-1 python keras_applications/predict.py --batch 48 -m mobilenet_v2 -n 5 Elephant.jpg --disable-eager
Elapsed 0.736 sec for 48 images. 65.228 images/sec
Elapsed 0.186 sec for 48 images. 257.518 images/sec
Elapsed 0.183 sec for 48 images. 262.571 images/sec
Elapsed 0.191 sec for 48 images. 251.067 images/sec
Elapsed 0.187 sec for 48 images. 256.843 images/sec
```

## Simple CNN tranining with mnist

```
% python mnist_cnn/mnist_cnn.py
batch_size=128 num_classes=10 epoch=12
x_train shape: (60000, 1, 28, 28)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 7.148695s 119us/sample - loss: 0.2166 - accuracy: 0.9348 - val_loss: 0.0596 - val_accuracy: 0.9814
Epoch 2/12
60000/60000 [==============================] - 6.480616s 108us/sample - loss: 0.0804 - accuracy: 0.9768 - val_loss: 0.0412 - val_accuracy: 0.9865
Epoch 3/12
60000/60000 [==============================] - 6.470301s 108us/sample - loss: 0.0620 - accuracy: 0.9818 - val_loss: 0.0424 - val_accuracy: 0.9856
Epoch 4/12
60000/60000 [==============================] - 6.511891s 109us/sample - loss: 0.0531 - accuracy: 0.9844 - val_loss: 0.0381 - val_accuracy: 0.9875
Epoch 5/12
60000/60000 [==============================] - 6.498200s 108us/sample - loss: 0.0504 - accuracy: 0.9857 - val_loss: 0.0338 - val_accuracy: 0.9894
Epoch 6/12
60000/60000 [==============================] - 6.496966s 108us/sample - loss: 0.0506 - accuracy: 0.9857 - val_loss: 0.0374 - val_accuracy: 0.9896
Epoch 7/12
60000/60000 [==============================] - 6.465539s 108us/sample - loss: 0.0510 - accuracy: 0.9854 - val_loss: 0.0325 - val_accuracy: 0.9902
Epoch 8/12
60000/60000 [==============================] - 6.474478s 108us/sample - loss: 0.0510 - accuracy: 0.9855 - val_loss: 0.0331 - val_accuracy: 0.9905
Epoch 9/12
60000/60000 [==============================] - 6.483711s 108us/sample - loss: 0.0514 - accuracy: 0.9856 - val_loss: 0.0341 - val_accuracy: 0.9899
Epoch 10/12
60000/60000 [==============================] - 6.457727s 108us/sample - loss: 0.0544 - accuracy: 0.9851 - val_loss: 0.0307 - val_accuracy: 0.9905
Epoch 11/12
60000/60000 [==============================] - 6.450509s 108us/sample - loss: 0.0527 - accuracy: 0.9859 - val_loss: 0.1054 - val_accuracy: 0.9846
Epoch 12/12
60000/60000 [==============================] - 6.452999s 108us/sample - loss: 0.0561 - accuracy: 0.9855 - val_loss: 0.0359 - val_accuracy: 0.9893
Test loss: 0.03589132128714991
Test accuracy: 0.98929995

% VE_NODE_NUMBER=-1 python mnist_cnn/mnist_cnn.py --nhwc
batch_size=128 num_classes=10 epoch=12
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 10.529186s 175us/sample - loss: 0.2283 - accuracy: 0.9308 - val_loss: 0.0573 - val_accuracy: 0.9799
Epoch 2/12
60000/60000 [==============================] - 9.762516s 163us/sample - loss: 0.0803 - accuracy: 0.9765 - val_loss: 0.0385 - val_accuracy: 0.9869
Epoch 3/12
60000/60000 [==============================] - 9.896653s 165us/sample - loss: 0.0630 - accuracy: 0.9809 - val_loss: 0.0342 - val_accuracy: 0.9891
Epoch 4/12
60000/60000 [==============================] - 9.889808s 165us/sample - loss: 0.0552 - accuracy: 0.9840 - val_loss: 0.0378 - val_accuracy: 0.9871
Epoch 5/12
60000/60000 [==============================] - 9.859850s 164us/sample - loss: 0.0515 - accuracy: 0.9857 - val_loss: 0.0377 - val_accuracy: 0.9877
Epoch 6/12
60000/60000 [==============================] - 9.711194s 162us/sample - loss: 0.0499 - accuracy: 0.9860 - val_loss: 0.0347 - val_accuracy: 0.9907
Epoch 7/12
60000/60000 [==============================] - 9.795707s 163us/sample - loss: 0.0491 - accuracy: 0.9863 - val_loss: 0.0307 - val_accuracy: 0.9901
Epoch 8/12
60000/60000 [==============================] - 9.839443s 164us/sample - loss: 0.0495 - accuracy: 0.9865 - val_loss: 0.0350 - val_accuracy: 0.9905
Epoch 9/12
60000/60000 [==============================] - 9.740287s 162us/sample - loss: 0.0510 - accuracy: 0.9863 - val_loss: 0.0387 - val_accuracy: 0.9888
Epoch 10/12
60000/60000 [==============================] - 9.930354s 166us/sample - loss: 0.0533 - accuracy: 0.9852 - val_loss: 0.0340 - val_accuracy: 0.9906
Epoch 11/12
60000/60000 [==============================] - 9.804379s 163us/sample - loss: 0.0515 - accuracy: 0.9855 - val_loss: 0.0347 - val_accuracy: 0.9897
Epoch 12/12
60000/60000 [==============================] - 9.928548s 165us/sample - loss: 0.0553 - accuracy: 0.9853 - val_loss: 0.0388 - val_accuracy: 0.9907
Test loss: 0.03884528672008182
Test accuracy: 0.9907
```

## MLP training with mnist

```
# VE(Type 10B)
% python mnist_mlp/mnist_mlp.py
Test loss: 0.09438222976020093
Test accuracy: 0.98209995
Elapsed time:   19.590 sec for 10 epochs.   30.629 Kimages/sec

# CPU(Xeon Gold 6126 x 2)
% VE_NODE_NUMBER=-1 python mnist_mlp/mnist_mlp.py
Test loss: 0.08251755181024363
Test accuracy: 0.9825
Elapsed time:   24.011 sec for 10 epochs.   24.988 Kimages/sec

# GPU(V100)
% python mnist_mlp/mnist_mlp.py
(snip)
Test loss: 0.10881402977931752
Test accuracy: 0.9798
Elapsed time:   24.676 sec for 10 epochs.   24.316 Kimages/sec
```
