import sys
import unittest

from keras_applications import predict
from mnist_cnn import mnist_cnn
from mnist_mlp import mnist_mlp

class Test(unittest.TestCase):
    def test_resnet50(self):
        args = "keras_applications/predict.py --batch 32 -m resnet50 -n 2 --nchw Elephant.jpg --disable-eager"
        with unittest.mock.patch.object(sys, 'argv', args.split()):
            time, result = predict.main()
            self.assertTrue(0.17 < time < 0.20)
            self.assertEqual(result[0][0][1], "African_elephant")
            self.assertAlmostEqual(result[0][0][2], 0.56663877)

    def test_mobilenetv2(self):
        args = "keras_applications/predict.py --batch 32 -m mobilenet_v2 -n 2 --nchw Elephant.jpg --disable-eager"
        with unittest.mock.patch.object(sys, 'argv', args.split()):
            time, result = predict.main()
            self.assertTrue(0.070 < time < 0.090)
            self.assertEqual(result[0][0][1], "tusker")
            self.assertAlmostEqual(result[0][0][2], 0.4446602)

    def test_mnist_cnn(self):
        args = "mnist_cnn/mnist_cnn.py -d /device:ve:0 --epoch 5"
        with unittest.mock.patch.object(sys, 'argv', args.split()):
            time, result = mnist_cnn.main()
            #print(time)
            #print(time, result)
            self.assertTrue(5.2 < time < 5.4)
            self.assertTrue(0.030 < result[0] < 0.040) # loss
            self.assertTrue(0.980 < result[1] < 0.999) # acculacy

    def test_mnist_mlp(self):
        args = "mnist_mlp/mnist_mlp.py -d /device:ve:0"
        with unittest.mock.patch.object(sys, 'argv', args.split()):
            time, result = mnist_mlp.main()
            #print(time, result)
            self.assertTrue(9.0 < time < 10.0)
            #self.assertTrue(0.080 < result[0] < 0.110) # loss
            self.assertTrue(0.980 < result[1] < 0.999) # acculacy


if __name__ == "__main__":
    unittest.main()


