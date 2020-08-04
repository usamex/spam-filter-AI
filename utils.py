def read_classification_from_file(fpath):
    dictionary = dict()
    with open(fpath, 'r') as fp:
        for line in fp:
            line_list = line.split()
            dictionary[line_list[0]] = line_list[1]
    return dictionary


def write_classification_to_file(cls_dict, fpath):
    with open(fpath, 'w') as fp:
        for i, j in cls_dict.items():
            line = ' '.join((i, j))
            line += '\n'
            fp.write(line)


if __name__ == '__main__':
    cls_dict = read_classification_from_file('1/!truth.txt')
    fpath = '1/!prediction.txt'
    write_classification_to_file(cls_dict, fpath)
