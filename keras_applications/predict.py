import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications

import argparse
import datetime

def load_image(img_path, batch=1):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.repeat(x[np.newaxis, :, :, :], batch, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nchw", action="store_true")
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--inter-threads", type=int)
    parser.add_argument("--disable-eager", action="store_true")
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("file")
    args = parser.parse_args()

    if args.nchw:
        tf.keras.backend.set_image_data_format('channels_first')
    else:
        tf.keras.backend.set_image_data_format('channels_last')

    if args.disable_eager :
      tf.compat.v1.disable_eager_execution()

    if args.inter_threads:
        print("inter_threads={}".format(args.inter_threads))
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    if args.model == "resnet50":
        preprocess_input = applications.resnet50.preprocess_input
        decode_predictions = applications.resnet50.decode_predictions
        model = applications.resnet.ResNet50(weights='imagenet')
    elif args.model == "mobilenet_v2":
        preprocess_input = applications.mobilenet_v2.preprocess_input
        decode_predictions = applications.mobilenet_v2.decode_predictions
        model = applications.mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    else:
        raise Exception("Use -m <resnet50|mobilenet_v2>")

    if args.verbose > 0:
        print("model: {}".format(args.model))

    if args.profile:
        log_dir="logs/"
        summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    img_path = args.file
    x = load_image(img_path, args.batch)
    x = preprocess_input(x)

    for n in range(args.n):
        if args.profile:
            tf.summary.trace_on(graph=True, profiler=True)

        start = time.time()
        preds = model.predict(x)
        elapsed = time.time() - start

        if args.profile:
            with summary_writer.as_default():
                tf.summary.trace_export(name="resnet50", step=n, profiler_outdir=log_dir)

        print("Elapsed {:.3f} sec for {} images. {:.3f} images/sec".format(elapsed, args.batch, args.batch / elapsed))

    if args.verbose > 0:
        tmp = decode_predictions(preds, top=3)
        for pred in tmp:
            print(pred)

if __name__ == "__main__":
    main()
