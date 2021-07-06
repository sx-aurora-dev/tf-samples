# TensorFlow-VE Samples

    % pip install Pillow
    % wget https://upload.wikimedia.org/wikipedia/commons/b/bc/Elephant.jpg

## ResNet50 inference

```
# VE (Type 10BE)
% python keras_applications/predict.py --batch 32 -m resnet50 -n 5 --nchw Elephant.jpg --disable-eager -v
Elapsed 1.071 sec for 32 images. 29.866 images/sec
Elapsed 0.181 sec for 32 images. 177.005 images/sec
Elapsed 0.182 sec for 32 images. 176.164 images/sec
Elapsed 0.182 sec for 32 images. 176.227 images/sec
Elapsed 0.180 sec for 32 images. 177.964 images/sec

# CPU (Xeon Gold 6226 x 2)
% python keras_applications/predict.py --batch 48 -m resnet50 -n 5 Elephant.jpg --disable-eager --nchw -d /cpu:0 -v
Elapsed 1.319 sec for 48 images. 36.398 images/sec
Elapsed 0.391 sec for 48 images. 122.717 images/sec
Elapsed 0.379 sec for 48 images. 126.777 images/sec
Elapsed 0.367 sec for 48 images. 130.651 images/sec
Elapsed 0.386 sec for 48 images. 124.502 images/sec


# GPU(V100S)
% python keras_applications/predict.py --batch 32 -m resnet50 -n 5 Elephant.jpg --disable-eager --nchw -d /gpu:0 -v
Elapsed 2.590 sec for 32 images. 12.356 images/sec
Elapsed 0.052 sec for 32 images. 614.474 images/sec
Elapsed 0.050 sec for 32 images. 635.711 images/sec
Elapsed 0.048 sec for 32 images. 662.297 images/sec
Elapsed 0.049 sec for 32 images. 658.973 images/sec
```

## MobileNetV2 inference

```
# VE (Type 10BE)
% python keras_applications/predict.py --batch 32 -m mobilenet_v2 -n 5 --nchw Elephant.jpg --disable-eager -v
Elapsed 0.849 sec for 32 images. 37.691 images/sec
Elapsed 0.081 sec for 32 images. 395.338 images/sec
Elapsed 0.080 sec for 32 images. 397.798 images/sec
Elapsed 0.082 sec for 32 images. 389.726 images/sec
Elapsed 0.083 sec for 32 images. 387.793 images/sec

# CPU (Xeon Gold 6226 x 2)
% python keras_applications/predict.py --batch 48 -m mobilenet_v2 -n 5 Elephant.jpg --disable-eager -v
Elapsed 0.941 sec for 48 images. 51.029 images/sec
Elapsed 0.170 sec for 48 images. 282.894 images/sec
Elapsed 0.168 sec for 48 images. 285.564 images/sec
Elapsed 0.167 sec for 48 images. 288.188 images/sec
Elapsed 0.168 sec for 48 images. 285.305 images/sec

# GPU(V100S)
% python keras_applications/predict.py --batch 32 -m mobilenet_v2 -n 5 Elephant.jpg --disable-eager --nchw -d /gpu:0 -v
Elapsed 2.376 sec for 32 images. 13.470 images/sec
Elapsed 0.030 sec for 32 images. 1069.225 images/sec
Elapsed 0.024 sec for 32 images. 1306.828 images/sec
Elapsed 0.025 sec for 32 images. 1304.427 images/sec
Elapsed 0.023 sec for 32 images. 1393.977 images/sec
```

## Simple CNN tranining with mnist

```
% python mnist_cnn/mnist_cnn.py -d /device:ve:0 -v
batch_size=128 num_classes=10 epoch=12
x_train shape: (60000, 1, 28, 28)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
469/469 [==============================] - 5.971250s 13ms/step - loss: 0.2163 - accuracy: 0.9345 - val_loss: 0.0492 - val_accuracy: 0.9839
Epoch 2/12
469/469 [==============================] - 5.926254s 13ms/step - loss: 0.0802 - accuracy: 0.9761 - val_loss: 0.0382 - val_accuracy: 0.9869
Epoch 3/12
469/469 [==============================] - 5.932469s 13ms/step - loss: 0.0630 - accuracy: 0.9816 - val_loss: 0.0362 - val_accuracy: 0.9885
Epoch 4/12
469/469 [==============================] - 5.948468s 13ms/step - loss: 0.0530 - accuracy: 0.9841 - val_loss: 0.0333 - val_accuracy: 0.9887
Epoch 5/12
469/469 [==============================] - 5.881706s 13ms/step - loss: 0.0509 - accuracy: 0.9856 - val_loss: 0.0366 - val_accuracy: 0.9895
Epoch 6/12
469/469 [==============================] - 5.933605s 13ms/step - loss: 0.0503 - accuracy: 0.9857 - val_loss: 0.0413 - val_accuracy: 0.9888
Epoch 7/12
469/469 [==============================] - 5.934583s 13ms/step - loss: 0.0489 - accuracy: 0.9864 - val_loss: 0.0319 - val_accuracy: 0.9905
Epoch 8/12
469/469 [==============================] - 5.958809s 13ms/step - loss: 0.0494 - accuracy: 0.9860 - val_loss: 0.0357 - val_accuracy: 0.9903
Epoch 9/12
469/469 [==============================] - 5.930326s 13ms/step - loss: 0.0520 - accuracy: 0.9852 - val_loss: 0.0321 - val_accuracy: 0.9911
Epoch 10/12
469/469 [==============================] - 5.879300s 13ms/step - loss: 0.0498 - accuracy: 0.9860 - val_loss: 0.0409 - val_accuracy: 0.9890
Epoch 11/12
469/469 [==============================] - 5.852392s 12ms/step - loss: 0.0504 - accuracy: 0.9859 - val_loss: 0.0315 - val_accuracy: 0.9912
Epoch 12/12
469/469 [==============================] - 5.919448s 13ms/step - loss: 0.0516 - accuracy: 0.9861 - val_loss: 0.0358 - val_accuracy: 0.9902

# CPU (Xeon Gold 6226 x 2)
% python mnist_cnn/mnist_cnn.py --nhwc -d /cpu:0 -v
batch_size=128 num_classes=10 epoch=12
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
469/469 [==============================] - 8.176388s 17ms/step - loss: 0.2219 - accuracy: 0.9333 - val_loss: 0.0527 - val_accuracy: 0.9834
Epoch 2/12
469/469 [==============================] - 8.089273s 17ms/step - loss: 0.0780 - accuracy: 0.9770 - val_loss: 0.0375 - val_accuracy: 0.9873
Epoch 3/12
469/469 [==============================] - 8.135950s 17ms/step - loss: 0.0622 - accuracy: 0.9817 - val_loss: 0.0333 - val_accuracy: 0.9888
Epoch 4/12
469/469 [==============================] - 8.034145s 17ms/step - loss: 0.0525 - accuracy: 0.9850 - val_loss: 0.0368 - val_accuracy: 0.9887
Epoch 5/12
469/469 [==============================] - 8.073792s 17ms/step - loss: 0.0505 - accuracy: 0.9854 - val_loss: 0.0296 - val_accuracy: 0.9906
Epoch 6/12
469/469 [==============================] - 8.005976s 17ms/step - loss: 0.0460 - accuracy: 0.9868 - val_loss: 0.0309 - val_accuracy: 0.9902
Epoch 7/12
469/469 [==============================] - 8.022778s 17ms/step - loss: 0.0424 - accuracy: 0.9875 - val_loss: 0.0304 - val_accuracy: 0.9903
Epoch 8/12
469/469 [==============================] - 7.979658s 17ms/step - loss: 0.0410 - accuracy: 0.9877 - val_loss: 0.0299 - val_accuracy: 0.9908
Epoch 9/12
469/469 [==============================] - 7.958190s 17ms/step - loss: 0.0417 - accuracy: 0.9877 - val_loss: 0.0296 - val_accuracy: 0.9908
Epoch 10/12
469/469 [==============================] - 8.003123s 17ms/step - loss: 0.0438 - accuracy: 0.9878 - val_loss: 0.0359 - val_accuracy: 0.9911
Epoch 11/12
469/469 [==============================] - 7.979754s 17ms/step - loss: 0.0421 - accuracy: 0.9879 - val_loss: 0.0358 - val_accuracy: 0.9911
Epoch 12/12
469/469 [==============================] - 7.992872s 17ms/step - loss: 0.0434 - accuracy: 0.9881 - val_loss: 0.0291 - val_accuracy: 0.9910
Test loss: 0.02940506301820278
Test accuracy: 0.9909999966621399

# GPU(V100S)
% python mnist_cnn/mnist_cnn.py -d /gpu:0 -v
469/469 [==============================] - 1.810667s 4ms/step - loss: 0.2069 - accuracy: 0.9370 - val_loss: 0.0500 - val_accuracy: 0.9847
Epoch 2/12
469/469 [==============================] - 1.564235s 3ms/step - loss: 0.0766 - accuracy: 0.9772 - val_loss: 0.0429 - val_accuracy: 0.9869
Epoch 3/12
469/469 [==============================] - 1.580783s 3ms/step - loss: 0.0604 - accuracy: 0.9822 - val_loss: 0.0324 - val_accuracy: 0.9891
Epoch 4/12
469/469 [==============================] - 1.537399s 3ms/step - loss: 0.0502 - accuracy: 0.9852 - val_loss: 0.0321 - val_accuracy: 0.9896
Epoch 5/12
469/469 [==============================] - 1.547240s 3ms/step - loss: 0.0466 - accuracy: 0.9865 - val_loss: 0.0373 - val_accuracy: 0.9876
Epoch 6/12
469/469 [==============================] - 1.577667s 3ms/step - loss: 0.0440 - accuracy: 0.9872 - val_loss: 0.0328 - val_accuracy: 0.9897
Epoch 7/12
469/469 [==============================] - 1.584175s 3ms/step - loss: 0.0428 - accuracy: 0.9882 - val_loss: 0.0432 - val_accuracy: 0.9877
Epoch 8/12
469/469 [==============================] - 1.596059s 3ms/step - loss: 0.0439 - accuracy: 0.9876 - val_loss: 0.0459 - val_accuracy: 0.9875
Epoch 9/12
469/469 [==============================] - 1.581552s 3ms/step - loss: 0.0444 - accuracy: 0.9872 - val_loss: 0.0369 - val_accuracy: 0.9892
Epoch 10/12
469/469 [==============================] - 1.603999s 3ms/step - loss: 0.0465 - accuracy: 0.9872 - val_loss: 0.0347 - val_accuracy: 0.9893
Epoch 11/12
469/469 [==============================] - 1.615616s 3ms/step - loss: 0.0482 - accuracy: 0.9864 - val_loss: 0.0346 - val_accuracy: 0.9905
Epoch 12/12
469/469 [==============================] - 1.593585s 3ms/step - loss: 0.0472 - accuracy: 0.9873 - val_loss: 0.0368 - val_accuracy: 0.9893
```

## MLP training with mnist

```
# VE(Type 10B)
% python mnist_mlp/mnist_mlp.py -d /device:ve:0 -v
Test loss: 0.0968351736664772
Test accuracy: 0.9824000000953674
Elapsed time:   14.048 sec for 10 epochs.   42.712 Kimages/sec

# CPU(Xeon Gold 6226 x 2)
% python mnist_mlp/mnist_mlp.py -d /cpu:0
Test loss: 0.08748051524162292
Test accuracy: 0.9817000031471252
Elapsed time:   20.203 sec for 10 epochs.   29.698 Kimages/sec

# GPU(V100S)
% python mnist_mlp/mnist_mlp.py -d /gpu:0
Test loss: 0.07711399346590042
Test accuracy: 0.9846000075340271
Elapsed time:    8.319 sec for 10 epochs.   72.126 Kimages/sec
```
