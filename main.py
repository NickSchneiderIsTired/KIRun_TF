import opensmile
from pipline import Pipeline
import tensorflow as tf
from model import Network
from utils import create_val_data

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Init necessary data
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=2
    )
    batch_size = 4
    dataset = Pipeline(smile,
                       "data/train/",
                       batch_size=batch_size,
                       num_parallel_calls=1)

    print("Creating training data")
    features, classification = create_val_data(smile, "data/train/")
    print("Creating validation data")
    val_x, val_y = create_val_data(smile, "data/val/")
    print("Data created")

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    net = Network()
    counter = 0
    '''
        for (filename, classification, features) in dataset():
            features = tf.reshape(features, (batch_size, 176))
            with tf.GradientTape() as tape:
                # noinspection PyCallingNonCallable
                res = net(features)
                # regularization = tf.add_n(net.losses)
                loss = tf.nn.softmax_cross_entropy_with_logits(classification, res)
                mean_loss = tf.reduce_mean(loss)

                if counter % 10 == 0:
                    correctly_classified_count = 0
                    print(mean_loss.numpy())
                    # noinspection PyCallingNonCallable
                    res = net(val_x)  # Calculate result of NN
                    for i in range(len(res)):
                        if tf.math.argmax(res[i]) == tf.math.argmax(val_y[i]):
                            correctly_classified_count += 1
                    accuracy = correctly_classified_count / len(res)
                    print("\nMODEL Accuracy: ", accuracy)

            grads = tape.gradient(mean_loss, net.trainable_weights)
            opt.apply_gradients(zip(grads, net.trainable_weights))
            counter += 1
        '''
    while True:
        indices = tf.range(start=0, limit=tf.shape(features)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_features = tf.gather(features, shuffled_indices)
        shuffled_classification = tf.gather(classification, shuffled_indices)
        with tf.GradientTape() as tape:
            # noinspection PyCallingNonCallable
            res = net(shuffled_features)
            # regularization = tf.add_n(net.losses)
            loss = tf.nn.softmax_cross_entropy_with_logits(shuffled_classification, res)
            mean_loss = tf.reduce_mean(loss)

            if counter % 100 == 0:
                correctly_classified_count = 0
                # noinspection PyCallingNonCallable
                res = net(val_x)  # Calculate result of NN
                for i in range(len(res)):
                    if tf.math.argmax(res[i]) == tf.math.argmax(val_y[i]):
                        correctly_classified_count += 1
                accuracy = correctly_classified_count / len(res)
                print("MODEL Accuracy: ", accuracy, '\n')
            if counter % 1000 == 0:
                print("Saving model")
                #  net.save('./trainedModel')

        grads = tape.gradient(mean_loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        counter += 1


if __name__ == '__main__':
    main()
