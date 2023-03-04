import argparse
import random
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "-n", type=int, default=20, help="Number of bins to place into")
    parser.add_argument("--picklists_path", "-pp", type=str, default=None, help="Path to pickable objects")
    parser.add_argument("--outfile_path", "-o", type = str, default = datetime.date.today().strftime("%d_%m_%Y_%H-%M-%S")+".txt", help = "output file path")
    args = parser.parse_args()

    picklists = []
    with open(args.picklists_path, "r") as infile:
        picklists = infile.read().split("\n")


    chosen_picklists = random.sample(picklists, args.n)

    with open(args.outfile_path, "w+") as outfile:
        for picklist in chosen_picklists:
                outfile.write(picklist + "\n")
    