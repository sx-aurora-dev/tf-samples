# Multi VE sample

Tested on two VEs system, VE0 and VE2 with CentOS8 and tf-ve 2.3.1.

Example from <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras>.

## Single VE

```
% python single.py
2021-05-11 10:52:26.530831: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2700000000 Hz
2021-05-11 10:52:26.535961: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564722cb5460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-05-11 10:52:26.535986: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch 1/3
70/70 [==============================] - 0.199116s 3ms/step - loss: 2.2544 - accuracy: 0.1953
Epoch 2/3
70/70 [==============================] - 0.192481s 3ms/step - loss: 2.1498 - accuracy: 0.4654
Epoch 3/3
70/70 [==============================] - 0.191019s 3ms/step - loss: 2.0412 - accuracy: 0.6056
```

## Multi VE

Important: Delete `https_proxy`.

```
% sh multi.sh
  :
70/70 [==============================] - 3.079076s 44ms/step - loss: 2.2906 - accuracy: 0.1234
Epoch 2/3
70/70 [==============================] - 2.993064s 43ms/step - loss: 2.2389 - accuracy: 0.2542
Epoch 3/3
70/70 [==============================] - 2.991904s 43ms/step - loss: 2.1783 - accuracy: 0.4124
```
