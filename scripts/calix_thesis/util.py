def parse_action_label_file(path_to_file):
    data = []
    with open(path_to_file, 'r') as infile:
        data = [line.strip().split(' ') for line in infile.readlines()]
        infile.close()
    keys = ['pick', 'carry_item', 'place', 'carry_empty']
    data_dict = dict()
    for key in keys:
        data_dict[key] = []
    for line in data:
        data_dict[line[0]].append([int(line[1]), int(line[2])])

    return data_dict