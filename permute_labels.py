import glob
import itertools
import numpy as np

labels = glob.glob("./5/lab/*")

words = []
picks = []
letters = []
print(labels[-1])
pick_permutations = [] # an array of sets each set is a permutation for a single pick
for label in labels:
    pick = []
    with open(label, 'r') as f:
        lines = f.readlines()[1:-1]
        word = ""
        for i in range(0, len(lines)):
            word += lines[i].replace("\n","")
        n = 4
        word_list = [word[i:i + n] for i in range(0, len(word), n)]
        permutations = list(itertools.permutations(word_list))
        print(set(permutations)) #repeat items result in redundant permutations
        pick_permutations.append(list(set(permutations)))

print(pick_permutations)
curr_permutations = []

# for element in itertools.product(pick_permutations[0], pick_permutations[1]):
#     print(element)
#
# for element in itertools.product(itertools.product(pick_permutations[0], pick_permutations[1]), pick_permutations[2]):
#     print(element)

product = itertools.product(pick_permutations[0], pick_permutations[1])
lis = [1,2,3,4]
print(lis[2:])
for pick_permutation in pick_permutations[2:]:
    product = itertools.product(pick_permutation, product)

elements = []
for element in product:
    elements.append(element)
print(len(elements))
print(type(elements[0]))
print(6**6)
