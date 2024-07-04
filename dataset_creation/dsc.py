from emoji import demojize
from transformers import BertTokenizer

# Normalize input string for further processing
def cosmetics(lines: str):
    lines = lines.replace('@@', '@').replace('\\u200b', '').replace('\\u003c3', '')
    lines = demojize(lines)
    lines = lines.replace('http://', 'httpurl').replace('https://', 'httpurl')
    lines = lines.replace('@', '_user')
    return lines

# Get the author from data entry
def get_author(lines: str):
    return lines.split('\t')[1].strip()

# Sort a dictionary by values
def sort_dict(dictionary: dict):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

# Format a dictionary to be saved as .tsv
def output_string(input_dict: dict):
    return '\n'.join([f"{key}\t{value}" for key, value in input_dict.items()])

# Split the comment into sentences with lower length
def len_split(comment: str, author: str, max_length: int):
    output = ''
    while len(comment.split()) > max_length:
        spl_comment = comment.split()
        split_pos = next((i for i, tok in enumerate(spl_comment) if tok in ['...', '.', '!', '?', 'and']), len(spl_comment))
        substring = ' '.join(spl_comment[:split_pos+1])
        if len(substring.split()) > max_length:
            substring = ' '.join(spl_comment[:max_length])
        output += f"{substring}\t{author}\n"
        comment = ' '.join(spl_comment[split_pos+1:])
    output += f"{comment}\t{author}\n"
    return output

# Export comments with word count greater than max_len
def overlen_export(path_to_file, max_len: int):
    with open(path_to_file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
    with open("E:\\research\\data_sets\\bak-files\\overlen.tsv", "a", encoding="utf-8") as output_file:
        for line in lines:
            if len(line.split()) > max_len:
                output_file.write(f"{line.strip()}\t{len(line.split())}\n")

# Export comments with token count greater than max_len using BERT tokenizer
def bert_overlen_export(path_to_file, max_len: int):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    with open(path_to_file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
    with open("E:\\research\\data_sets\\bak-files\\bert_overlen.tsv", "a", encoding="utf-8") as output_file:
        for line in lines:
            comment, author = line.split('\t', 1)
            if len(tokenizer.tokenize(comment)) > max_len:
                output_file.write(f"{comment}\t{author}")

# Clean the dataset from overlength comments
def dataset_overlen_cleaner(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as source_file:
        lines = source_file.readlines()
    with open("E:\\research\\data_sets\\bak-files\\bert_overlen.tsv", "r", encoding="utf-8") as dump_file:
        dumps = set(dump_file.readlines())
    with open("E:\\research\\data_sets\\bak-files\\dataset_clean.tsv", "a", encoding="utf-8") as output_file:
        for line in lines:
            if line not in dumps:
                output_file.write(line)
