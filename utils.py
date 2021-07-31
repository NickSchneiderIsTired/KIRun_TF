from os import listdir
import audiofile
import numpy as np
import random
import tensorflow as tf


class AnswerSet:
    def __init__(self, filename, wealth, burden, ground, running_start, running_stop):
        self.filename = filename
        self.wealth = wealth
        self.burden = burden
        self.ground = ground
        self.running_start = running_start
        self.running_stop = running_stop

    def __call__(self):
        return self.wealth, self.burden, self.ground, self.running_start, self.running_stop

    def __array__(self):
        return np.array([self.wealth, self.burden, self.running_start, self.running_stop])

    @staticmethod
    def as_array(filename, wealth, burden, ground, running_start, running_stop):
        return np.array([filename, wealth, burden, int(running_start), int(running_stop)])


# Process chunk of audio file
def read_chunk(smile, file, start, duration):
    signal, sampling_rate = audiofile.read(file, always_2d=True, offset=start, duration=duration)
    process = smile.process_signal(
        signal,
        sampling_rate
    )
    return process.values


#  Filter answers from annotation file
def read_groundtruth(file):
    with open(file) as f:
        answers = []
        w, b, g = 0, 0, 0
        for line in f.readlines():
            # Annotation
            answer_string = line.split()[2]

            # Save current dataset and reset data if walking sequence
            if answer_string[:1] == 'l':
                rows = line.split()
                answers.append(AnswerSet.as_array(
                    file.split('.')[0] + '.wav', int(w), int(b), g, float(rows[0]), float(rows[1])))
                w, b, g = 0, 0, 0
                continue

            # Parse answers
            sequence = answer_string[:2]
            if sequence == 'aw':
                w = answer_string[3:]
            elif sequence == 'ab':
                b = answer_string[3:]
            elif sequence == 'au':
                g = answer_string[3:]
        return np.array(answers)


#  Create array of wav filenames with according groundtruth
def create_samples(path):
    samples = np.empty((0, 5))
    for file in listdir(path):
        if file.endswith('.wav'):
            answer_set = read_groundtruth(path + file.split('.')[0] + '.txt')
            samples = np.append(samples, answer_set, axis=0)

    return samples


def classify_groundtruth(answer_set):
    # wealth = answer_set[0]
    burden = answer_set[1]
    return np.array([burden > 11], dtype="float32")


def create_data(smile, path):
    samples = create_samples(path)
    x_val = []
    y_val = []
    for sample in samples:
        filename = sample[0]
        answer_set = sample[1:4].astype(int)
        start_stamp = answer_set[-2]  # Start
        features = read_chunk(smile, filename, start_stamp, 5)
        classification = classify_groundtruth(answer_set)
        x_val.append(features)
        y_val.append(classification)
    return tf.reshape(tf.convert_to_tensor(x_val), (len(x_val), 176)), tf.convert_to_tensor(y_val)
