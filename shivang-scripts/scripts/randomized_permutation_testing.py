"""

one-tailed p test that gives p value of sample1 mean > sample2 mean.
this versions tests only some number of runs instead of all the permutations

"""
import numpy as np
from math import factorial

SAMPLE1_FILE = "outputs/aruco_only_gaussian_forward_filling_rms.txt"
SAMPLE2_FILE = "outputs/hsv_only_rms.txt"

NUM_RUNS = 100000

def read_samples(file_name):

    with open(file_name) as infile:
        return [float(i) for i in infile.readlines()]


sample1 = read_samples(SAMPLE1_FILE)
sample2 = read_samples(SAMPLE2_FILE)

all_samples = np.array(sample1 + sample2)

print (f"sample1 : {sample1}")
print (f"sample2 : {sample2}")
print (f"all samples: {all_samples}")

# sanity check to ensure there are sufficient permutations in the first place
if factorial(len(all_samples)) / factorial(len(sample1)) / factorial(len(sample2)) <= NUM_RUNS:
    raise ValueException("too many runs (NUM_RUNS) for sample, insufficient permutations ")

sample_difference = np.mean(sample1) - np.mean(sample2)

print (f"sample_difference: {sample_difference}")

total_num_samples = len(sample1) + len(sample2)

# calculate the total of all the samples
total_all_samples = np.sum(sample1) + np.sum(sample2)

# store the permutationst that have been used before so we do not use them again
permutations_used = set()

num_less_than = 0


for _ in range(NUM_RUNS):
    # choose the sample from all samples for the permutation test, record the indices so we can ensure we don't reuse
    # the same sample
    test_sample1_indices = np.random.choice(total_num_samples, len(sample1), replace=False)

    test_sample1_indices_tuple = tuple(sorted(test_sample1_indices))

    while test_sample1_indices_tuple in permutations_used:
        # keep resampling until a different permutation is obtained
        test_sample1_indices = np.random.choice(total_num_samples, len(sample1), replace=False)

        test_sample1_indices_tuple = tuple(sorted(test_sample1_indices))

    # compute the reverse since we might have the case where number of samples in sample1 == sample2 so it would be the
    # same sample just in "reverse" order
    test_sample2_indices_tuple = tuple([i for i in range(total_num_samples) if i not in test_sample1_indices])

    permutations_used.add(test_sample1_indices_tuple)
    permutations_used.add(test_sample2_indices_tuple)

    test_sample1 = all_samples[test_sample1_indices]
    
    test_sample1_mean = np.mean(test_sample1)
    # take total of all samples and minus the values in first sample which would be sum of the second sample then divide
    # by number of elements in the second sample
    test_sample2_mean = (total_all_samples - test_sample1_mean * len(sample1)) / len(sample2)

    if test_sample1_mean - test_sample2_mean <= sample_difference:
        num_less_than += 1

print (f"num_less_than: {num_less_than}")

print (f"p-value: {num_less_than / NUM_RUNS}")
