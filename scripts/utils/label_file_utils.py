

'''
Given a file with bins and items (describing a picklist), return in dict form.
'''
def label_to_dict(file_path):
    ans = {1: [], 2: [], 3: []}
    with open(file_path, "r") as f:
        lines = f.readlines()

        #change this if our raw.txt structure changes
        data = lines[0]
        data = [data[i:i+2] for i in range(0, len(data), 2)]
        for pick in data:
            ans[int(pick[1])].append(pick[0])
    
    return ans 