import matplotlib.pyplot as plt


with open("rmse-oop-2picklists.txt", 'r') as infile:
    lines = [i.split() for i in infile.readlines()]
    out_of_place = []
    rmse = []
    for i,j in lines:
        out_of_place.append(int(j))
        rmse.append(float(i))

plt.scatter(out_of_place, rmse)
plt.show()
