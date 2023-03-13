import argparse
import datetime
import random
import itertools
from collections import Counter
from functools import reduce

#I'm so mad that we don't have a built in multinomial function
fact = [0, 1]

#we expect a list of tuples that define a pick-place pair.
def get_permutations(picklist: list) -> int:
    count_vals = list(Counter(picklist).values())
    n = len(picklist)

    perms = fact[n]

    for val in count_vals:
        perms /= fact[val]

    return perms

"""
COLORS in colors.txt:
r, g, b, p, q, o, s, a, t, u
red, green, blue, light blue, light green, orange, alligator clip, yellow, clear, candle
"""

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", "-b", type=int, default=3, help="Number of bins to place into")
    parser.add_argument("--pick_items", "-pi", type=str, default=None, help="Path to pickable objects")
    parser.add_argument("--max_permutations", "-mp", type=int, default=12, help="Maximum number of permutations to allow.")
    parser.add_argument("--outfile_path", "-o", type = str, default = datetime.date.today().strftime("%d_%m_%Y_%H-%M-%S")+".txt", help = "output file path")
    parser.add_argument("--max_number", "-mn", type=int, default = 4, help="Maximum number of picks")
    args = parser.parse_args()

    pick_items = []
    with open(args.pick_items, "r") as infile:
        pick_items = [_ for _ in infile.readline().split(', ')] #we expect comma-separated items

    bins = range(1, args.bins + 1)
    pick_place_combos = list(itertools.product(pick_items, bins))

    legal_picklists = []


    i = 1
    while i <= args.max_number:
        print(f"Generating picklists with {i} picks.")
        #extend our factorial array for math
        if i > 1:
            fact.append(i * fact[i - 1])

        combos_i = list(itertools.combinations_with_replacement(pick_place_combos, i))
        filtered_combos_i = list(filter(lambda x: get_permutations(x) <= args.max_permutations, combos_i))
        if len(filtered_combos_i) == 0:
            break
        legal_picklists.extend(filtered_combos_i)
        i += 1    

    if args.outfile_path:
        with open(args.outfile_path, "w+") as outfile:
            for picklist in legal_picklists:
                picklist_line = ''.join([f"{item[0]}{item[1]}" for item in picklist])
                outfile.write(picklist_line + "\n")
