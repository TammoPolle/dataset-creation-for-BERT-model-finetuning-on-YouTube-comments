#  evaluation helper functions
from dataset_creation import dsc


#  get the comments for a list of labels from the source file
def get_comments(source_path: str, labels: list):
    lines = open(source_path, 'r', encoding='utf-8').readlines()
    comment_list = []
    for label in labels:
        for line in lines:
            if str(label) == dsc.get_author(line):
                comment_list.append(line)
    return comment_list


def reduce(source_path: str, target_path: str, blacklist: list):
    lines = open(source_path, 'r', encoding='utf-8').readlines()
    lines.remove(lines[0])
    with open(target_path, 'a', encoding='utf-8') as output_file:
        output_file.write('Comment\tAuthor\n')
        for line in lines:
            if int(dsc.get_author(line)) not in blacklist:
                if line[:line.find(dsc.get_author(line))] != "" and line[:line.find(dsc.get_author(line))] != "\t":
                    output_file.write(line)
