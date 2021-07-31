import opensmile
import tensorflow as tf
from model import Network
from utils import create_data

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

    print("Creating training data")
    features, classification = create_data(smile, "newData/train/")
    print("Creating validation data")
    val_x, val_y = create_data(smile, "newData/val/")
    print("Data created")

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    net = Network()
    counter = 0
    bestAccuracy = 0.0
    while True:
        indices = tf.range(start=0, limit=tf.shape(features)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_features = tf.gather(features, shuffled_indices)
        shuffled_classification = tf.gather(classification, shuffled_indices)
        with tf.GradientTape() as tape:
            # noinspection PyCallingNonCallable
            res = net(shuffled_features)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(shuffled_classification, res)
            regularization = tf.add_n(net.losses)
            mean_loss = tf.reduce_mean(loss + regularization)

            if counter % 100 == 0:
                print("Train loss: ", mean_loss.numpy())
                # noinspection PyCallingNonCallable
                val_res = net(val_x)
                val_loss = tf.nn.sigmoid_cross_entropy_with_logits(val_y, val_res)
                val_mean_loss = tf.reduce_mean(val_loss + tf.add_n(net.losses))
                print("Val loss: ", val_mean_loss.numpy())
                val_res = tf.math.sigmoid(val_res)  # Calculate result of NN
                val_loss = tf.math.abs(tf.cast(val_res > 0.5, dtype=tf.float32) - val_y)
                accuracy = 1 - tf.reduce_sum(val_loss) / tf.size(val_loss, out_type=tf.dtypes.float32)
                print("Model accuracy: ", accuracy.numpy(), '\n')
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    print("Saving model")
                    confusion_matrix = tf.math.confusion_matrix(tf.reshape(val_y, 23), tf.cast(tf.reshape(val_res, 23) > 0.5, dtype=tf.float32))
                    print(confusion_matrix.numpy())
                    # net.save('./trainedModel')
                    return 0

        grads = tape.gradient(mean_loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        counter += 1


if __name__ == '__main__':
    main()
