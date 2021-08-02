from absl import app, flags
import audiofile
import opensmile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('file', None, 'path to audio file of runner sample (required)')
flags.DEFINE_integer('timestamp', 0, 'timestamp in seconds where to start the 5 second sample at (default: 0)')


def read_chunk(file, start):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=2
    )
    signal, sampling_rate = audiofile.read(file, always_2d=True, offset=start, duration=5)
    process = smile.process_signal(
        signal,
        sampling_rate
    )
    return process.values


def main(argv):
    warnings.filterwarnings('ignore')
    file = FLAGS.file
    timestamp = FLAGS.timestamp
    if file is None:
        print("No file specified")
        return -1
    x = read_chunk(file, timestamp)
    net = tf.keras.models.load_model('./trainedModel', compile=False)
    y_pred = net(tf.convert_to_tensor(x))
    result = tf.cast(tf.math.sigmoid(y_pred) > 0.5, dtype=tf.int32)[0][0]
    print("High burden" if result == 1 else "Low burden")
    return 0


if __name__ == '__main__':
    app.run(main)

