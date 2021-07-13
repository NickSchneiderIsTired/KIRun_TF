from utils import create_samples, read_chunk, classify_groundtruth
import tensorflow as tf
import random


class Pipeline:
    def __init__(self, smile, path_to_data, batch_size, num_parallel_calls):
        self.smile = smile
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.samples = create_samples(path_to_data)

    def data_gen(self):
        for sample in self.samples:
            yield sample[0], sample[1:4]

    def __call__(self):
        dataset = tf.data.Dataset.from_generator(self.data_gen, output_types=(tf.string, tf.int32))
        dataset = dataset.shuffle(buffer_size=len(self.samples))
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda filename, answer_set: tf.numpy_function(self.load_single_example, (filename, answer_set),
                                                           (tf.string, tf.int32, tf.float32)),
            num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def load_single_example(self, filename, answer_set):
        # Get random 5 seconds from running sequence
        start_stamp = answer_set[-2]  # Start
        end_stamp = answer_set[-1]  # Stop
        start = random.randint(start_stamp, end_stamp - 5)
        features = read_chunk(self.smile, filename, start, 5)
        classification = classify_groundtruth(answer_set)
        return filename, classification, features
