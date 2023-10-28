import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

with open ('lists.txt', 'r') as f:
    lists = f.readlines()

fin_list = []
for pick in lists:
    pick = pick[:-1]
    # print(pick)
    pick_len = len(pick)
    # print(pick_len)
    cur_order = []
    for i in range(0,8,2):
        # print(pick[i])
        try:
            cur_order.append(pick[i])
        except:
            cur_order.append('')
    # print(cur_order)
    fin_list.append(cur_order)

# print(fin_list)

fin_list = np.array(fin_list)
print(fin_list.shape)
cur_class = 'r'
a = fin_list[:,0]
b = fin_list[:,1]
c = fin_list[:,2]
d = fin_list[:,3]
class_labels = []
for idx in range(fin_list.shape[0]):
    if a[idx] == cur_class or b[idx] == cur_class or c[idx] == cur_class or d[idx] == cur_class:
        class_labels.append(1)
    else:
        class_labels.append(0)

# print(fin_list[:,0])

df = pd.DataFrame()
df['a'] = a
df['b'] = b
df['c'] = c
df['d'] = d
df['label'] = class_labels

print(df.head())

data = df.values

print(data)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(data[:,:-1], data[:,-1])


# text_representation = tree.export_text(clf)
# print(text_representation)

# a = []
# b = []
# c = []
# d = []
# label = []

# for pick in fin_list:
#     for 