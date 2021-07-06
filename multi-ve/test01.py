import tensorflow as tf


def main():
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    def replica_fn():
        print('replica_fn')

    strategy.run(replica_fn)


if __name__ == "__main__":
    main()
