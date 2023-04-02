"""

one-tailed p test that gives p value of sample1 mean > sample2 mean.
this versions generates all permutations and tests them

"""
import numpy as np
from math import factorial

SAMPLE1_FILE = "outputs/aruco_only_rms.txt"
SAMPLE2_FILE = "outputs/hsv_only_rms.txt"

def read_samples(file_name):

    with open(file_name) as infile:
        return [float(i) for i in infile.readlines()]

def generate_permutations(elements, curr_array, curr_index, target_length):
    print ((elements, curr_array, curr_index, target_length))
    if len(curr_array) == target_length:
        # nothing left to do since already reached the target length, generate a new array that isn't affected by others too
        return [[i for i in curr_array]]

    if curr_index >= len(elements):
        # nothing left to add so this is invalid, do this check after seeing if curr_array is valid, otherwise might reject
        # some valid answers just cause curr_index is out of bounds
        return []

    else:
        # we can either include the current element or don't include it
        curr_array.append(elements[curr_index])
        include_curr = generate_permutations(elements, curr_array, curr_index + 1, target_length)

        # don't include
        curr_array.pop()
        no_include_curr = generate_permutations(elements, curr_array, curr_index + 1, target_length)

        return include_curr + no_include_curr
    

sample1 = read_samples(SAMPLE1_FILE)
sample2 = read_samples(SAMPLE2_FILE)

all_samples = np.array(sample1 + sample2)

print (f"sample1 : {sample1}")
print (f"sample2 : {sample2}")
print (f"all samples: {all_samples}")

sample_difference = np.mean(sample1) - np.mean(sample2)

total_num_samples = len(sample1) + len(sample2)

# calculate the total of all the samples
total_all_samples = np.sum(sample1) + np.sum(sample2)

# store the permutationst that have been used before so we do not use them again
permutations_used = set()

sample_indices = generate_permutations([i for i in range(total_num_samples)], [], 0, len(sample1))

num_greater_than = 0

counter = 0

for test_sample1_indices in sample_indices:

    if counter % 1000 == 0:
        print (counter)

    test_sample1_indices_tuple = tuple(sorted(test_sample1_indices))

    if test_sample1_indices_tuple in permutations_used:
        # get new permutation (might happen if sample1 and sample2 are same length then sample2 can be substituted for
        # sample 1
        continue

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

    if test_sample1_mean - test_sample2_mean >= sample_difference:
        num_greater_than += 1

print (f"num_greater_than: {num_greater_than}")

    
