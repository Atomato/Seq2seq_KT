import csv
import os

import numpy as np
from sklearn.utils import shuffle


def pad(data, target_length, target_value=0):
    return np.pad(data, (0, target_length - len(data)), 'constant', constant_values=target_value)


def one_hot(indices, depth):
    encoding = np.concatenate((np.eye(depth), [np.zeros(depth)]))
    return encoding[indices]


class OriginalInputProcessor(object):
    def process_problems_and_corrects(self, problem_seqs, correct_seqs, num_problems, is_encode,
                                      is_train=True, is_entire_sequence=False):
        """
        This function aims to process the problem sequence and the correct sequence into a DKT feedable X and y.
        :param problem_seqs: it is in shape [batch_size, None]
        :param correct_seqs: it is the same shape as problem_seqs
        :return:
        """
        # pad the sequence with the maximum sequence length
        max_seq_length = max([len(problem) for problem in problem_seqs])
        problem_seqs_pad = np.array(
            [pad(problem, max_seq_length, target_value=-1) for problem in problem_seqs])
        correct_seqs_pad = np.array(
            [pad(correct, max_seq_length, target_value=-1) for correct in correct_seqs])

        # find the correct seqs matrix as the following way:
        # Let problem_seq = [1,3,2,-1,-1] as a and correct_seq = [1,0,1,-1,-1] as b, which are padded already
        # First, find the element-wise multiplication of a*b*b = [1,0,2,-1,-1]
        # Then, for any values 0, assign it to -1 in the vector = [1,-1,2,-1,-1] as c
        # Such that when we one hot encoding the vector c, it will results a zero vector
        # temp is c in the comment.
        temp = problem_seqs_pad * correct_seqs_pad * correct_seqs_pad
        temp[temp == 0] = -1
        correct_seqs_pad = temp

        # one hot encode the information
        problem_seqs_oh = one_hot(problem_seqs_pad, depth=num_problems)
        correct_seqs_oh = one_hot(correct_seqs_pad, depth=num_problems)

        # slice out the x and y
        if is_entire_sequence:
            if is_train:
                x_problem_seqs = problem_seqs_oh[:, :-1]
                x_correct_seqs = correct_seqs_oh[:, :-1]
                y_problem_seqs = problem_seqs_oh[:, 1:]
                y_correct_seqs = correct_seqs_oh[:, 1:]
            else:
                x_problem_seqs = problem_seqs_oh[:, :]
                x_correct_seqs = correct_seqs_oh[:, :]
                y_problem_seqs = problem_seqs_oh[:, :]
                y_correct_seqs = correct_seqs_oh[:, :]

            X = np.concatenate((x_problem_seqs, x_correct_seqs), axis=2)

            result = (X, y_problem_seqs, y_correct_seqs)
            return result
        else:
            if is_train:
                x_problem_seqs = problem_seqs_oh[:, :]
                x_correct_seqs = correct_seqs_oh[:, :]
            else:
                x_problem_seqs = problem_seqs_oh[:, :]
                x_correct_seqs = correct_seqs_oh[:, :]

            if is_encode:
                X = np.concatenate((x_problem_seqs, x_correct_seqs), axis=2)
                return X
            else:
                return (x_problem_seqs, x_correct_seqs)

    def process_masks(self, encode_problem_seqs, decode_problem_seqs, num_problems):
        mask_seqs = []
        for enc_prob_seq, dec_prob_seq in zip(encode_problem_seqs, decode_problem_seqs):
            mask = [-1] * len(enc_prob_seq) + dec_prob_seq
            mask_seqs.append(mask)

        # pad the sequence with the maximum sequence length
        max_seq_length = max([len(problem) for problem in mask_seqs])
        mask_seqs_pad = np.array(
            [pad(problem, max_seq_length, target_value=-1) for problem in mask_seqs])

        # one hot encode the information
        mask_seqs_oh = one_hot(mask_seqs_pad, depth=num_problems)

        return mask_seqs_oh[:, 1:]


class BatchGenerator:
    """
    Generate batch for DKT model
    """

    def __init__(self, problem_seqs, correct_seqs,
                 encode_problem_seqs, encode_correct_seqs, decode_problem_seqs, decode_correct_seqs,
                 num_problems, batch_size, input_processor=OriginalInputProcessor(), **kwargs):
        self.cursor = 0  # point to the current batch index
        self.problem_seqs = problem_seqs
        self.correct_seqs = correct_seqs
        self.encode_problem_seqs = encode_problem_seqs
        self.encode_correct_seqs = encode_correct_seqs
        self.decode_problem_seqs = decode_problem_seqs
        self.decode_correct_seqs = decode_correct_seqs
        self.batch_size = batch_size
        self.num_problems = num_problems
        self.num_samples = len(encode_problem_seqs)
        self.num_batches = len(encode_problem_seqs) // batch_size + 1
        self.input_processor = input_processor
        self._current_batch = None

    def next_batch(self, is_train=True, is_entire_sequence=False, is_mask=False):
        start_idx = self.cursor * self.batch_size
        end_idx = min((self.cursor + 1) * self.batch_size, self.num_samples)
        problem_seqs = self.problem_seqs[start_idx:end_idx]
        correct_seqs = self.correct_seqs[start_idx:end_idx]
        encode_problem_seqs = self.encode_problem_seqs[start_idx:end_idx]
        encode_correct_seqs = self.encode_correct_seqs[start_idx:end_idx]
        decode_problem_seqs = self.decode_problem_seqs[start_idx:end_idx]
        decode_correct_seqs = self.decode_correct_seqs[start_idx:end_idx]

        if is_entire_sequence:
            # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
            self._current_batch = self.input_processor.process_problems_and_corrects(problem_seqs,
                                                                                     correct_seqs,
                                                                                     self.num_problems,
                                                                                     is_encode=True,
                                                                                     is_train=is_train,
                                                                                     is_entire_sequence=is_entire_sequence)

            if is_mask:
                mask_seqs = self.input_processor.process_masks(
                    encode_problem_seqs, decode_problem_seqs, self.num_problems)

                self._update_cursor()
                return self._current_batch, mask_seqs

        else:
            # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
            X = self.input_processor.process_problems_and_corrects(encode_problem_seqs,
                                                                   encode_correct_seqs,
                                                                   self.num_problems,
                                                                   is_encode=True,
                                                                   is_train=is_train)

            # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
            y_problem_seqs, y_correct_seqs = \
                self.input_processor.process_problems_and_corrects(decode_problem_seqs,
                                                                   decode_correct_seqs,
                                                                   self.num_problems,
                                                                   is_encode=False,
                                                                   is_train=is_train)

            self._current_batch = (X, y_problem_seqs, y_correct_seqs)

        self._update_cursor()
        return self._current_batch

    @property
    def current_batch(self):
        if self._current_batch is None:
            print("Current batch is None.")
        return None

    def _update_cursor(self):
        self.cursor = (self.cursor + 1) % self.num_batches

    def reset_cursor(self):
        self.cursor = 0

    def shuffle(self):
        self.problem_seqs, self.correct_seqs, \
            self.encode_problem_seqs, self.encode_correct_seqs, \
            self.decode_problem_seqs,  self.decode_correct_seqs = shuffle(
                self.problem_seqs, self.correct_seqs,
                self.encode_problem_seqs, self.encode_correct_seqs,
                self.decode_problem_seqs,  self.decode_correct_seqs, random_state=42)


def read_data_from_csv(filename):
    # read the csv file
    rows = []
    with open(filename, 'r') as f:
        print("Reading {0}".format(filename))
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
        print("{0} lines was read".format(len(rows)))

    # tuples stores the student answering sequence as
    # ([num_problems_answered], [problem_ids], [is_corrects])
    max_seq_length = 0
    num_problems = 0
    tuples = []
    for i in range(0, len(rows), 3):
        # numbers of problem a student answered
        seq_length = len(rows[i + 1])

        # only keep student with at least 6 records.
        if seq_length < 6:
            continue

        problem_seq = rows[i + 1]
        correct_seq = rows[i + 2]

        invalid_ids_loc = [i for i, pid in enumerate(problem_seq) if pid == '']
        for invalid_loc in invalid_ids_loc:
            del problem_seq[invalid_loc]
            del correct_seq[invalid_loc]

        # convert the sequence from string to int.
        problem_seq = list(map(int, problem_seq))
        correct_seq = list(map(int, correct_seq))

        # slice out the encode and decode sequence
        slice_idx = (seq_length - 1) // 2
        encode_problem_seq = problem_seq[:slice_idx]
        encode_correct_seq = correct_seq[:slice_idx]
        decode_problem_seq = problem_seq[slice_idx:]
        decode_correct_seq = correct_seq[slice_idx:]

        tup = (seq_length, problem_seq, correct_seq,
               encode_problem_seq, encode_correct_seq,
               decode_problem_seq, decode_correct_seq)
        tuples.append(tup)

        if max_seq_length < seq_length:
            max_seq_length = seq_length

        pid = max(int(pid) for pid in problem_seq if pid != '')
        if num_problems < pid:
            num_problems = pid
    # add 1 to num_problems because 0 is in the pid
    num_problems += 1

    print("max_num_problems_answered:", max_seq_length)
    print("num_problems:", num_problems)
    print("The number of data is {0}".format(len(tuples)))
    print("Finish reading data.")

    return tuples, num_problems, max_seq_length


class DKTData:
    def __init__(self, train_path, valid_path, test_path, batch_size=32):
        self.students_train, num_problems_train, max_seq_length_train = read_data_from_csv(
            train_path)
        self.students_valid, num_problems_valid, max_seq_length_valid = read_data_from_csv(
            valid_path)
        self.students_test, num_problems_test, max_seq_length_test = read_data_from_csv(
            test_path)
        self.num_problems = max(
            num_problems_test, num_problems_train, num_problems_valid)
        self.max_seq_length = max(
            max_seq_length_train, max_seq_length_test, max_seq_length_valid)

        problem_seqs = [student[1] for student in self.students_train]
        correct_seqs = [student[2] for student in self.students_train]
        encode_problem_seqs = [student[3] for student in self.students_train]
        encode_correct_seqs = [student[4] for student in self.students_train]
        decode_problem_seqs = [student[5] for student in self.students_train]
        decode_correct_seqs = [student[6] for student in self.students_train]
        self.train = BatchGenerator(
            problem_seqs, correct_seqs,
            encode_problem_seqs, encode_correct_seqs, decode_problem_seqs, decode_correct_seqs,
            self.num_problems, batch_size)

        problem_seqs = [student[1] for student in self.students_valid]
        correct_seqs = [student[2] for student in self.students_valid]
        encode_problem_seqs = [student[3] for student in self.students_valid]
        encode_correct_seqs = [student[4] for student in self.students_valid]
        decode_problem_seqs = [student[5] for student in self.students_valid]
        decode_correct_seqs = [student[6] for student in self.students_valid]
        self.valid = BatchGenerator(
            problem_seqs, correct_seqs,
            encode_problem_seqs, encode_correct_seqs, decode_problem_seqs, decode_correct_seqs,
            self.num_problems, batch_size)

        problem_seqs = [student[1] for student in self.students_test]
        correct_seqs = [student[2] for student in self.students_test]
        encode_problem_seqs = [student[3] for student in self.students_test]
        encode_correct_seqs = [student[4] for student in self.students_test]
        decode_problem_seqs = [student[5] for student in self.students_test]
        decode_correct_seqs = [student[6] for student in self.students_test]
        self.test = BatchGenerator(
            problem_seqs, correct_seqs,
            encode_problem_seqs, encode_correct_seqs, decode_problem_seqs, decode_correct_seqs,
            self.num_problems, batch_size)
