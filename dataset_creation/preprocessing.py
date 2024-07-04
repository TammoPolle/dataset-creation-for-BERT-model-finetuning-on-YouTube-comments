import os
import random
import dsc

# Verzeichnispfade
source_dir = 'E:\\research\\datapool\\raw\\combined\\'
target_dir = 'E:\\research\\datapool\\preprocessed\\combined\\min_len\\'

pseudonym_dict = {}
complete_auth_dis = {}
max_length = 126
min_length = 10


def process_file(filename):
    temp_auth_dis = {}
    with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as input_file:
        source_data = input_file.readlines()[1:]  # Überspringe die erste Zeile (Header)

    output_lines = []
    for line in source_data:
        comment, author = process_line(line)
        if author:
            if author not in pseudonym_dict:
                pseudonym_dict[author] = random.choice(range(1, 10000000))
            pseudo_author = str(pseudonym_dict[author])
            line = line.replace(author, pseudo_author)
            complete_auth_dis[pseudo_author] = complete_auth_dis.get(pseudo_author, 0) + 1
            temp_auth_dis[pseudo_author] = temp_auth_dis.get(pseudo_author, 0) + 1
        line = dsc.cosmetics(line)
        if validate_line(line):
            output_lines.append(line)
        else:
            split_comments = dsc.len_split(line.split('\t')[0], pseudo_author, max_length)
            for comment in split_comments:
                output_lines.append(f"{comment}\t{pseudo_author}")
                complete_auth_dis[pseudo_author] += 1
                temp_auth_dis[pseudo_author] += 1

    save_output(output_lines, filename, temp_auth_dis)


def process_line(line):
    for _ in range(6):
        line = line[line.find('\t') + 1:]
    for _ in range(7):
        line = line[:line.rfind('\t')]
    author = dsc.get_author(line)
    return line, author


def validate_line(line):
    parts = line.split('\t')
    return len(parts) == 2 and min_length < len(parts[0].split()) < max_length


def save_output(output_lines, filename, temp_auth_dis):
    base_filename = filename[filename.find('_') + 1:filename.find('.bak')]
    complete_output_path = os.path.join(target_dir, 'n-p_preprocessed_min_len.tsv')
    channel_output_path = os.path.join(target_dir, f"{base_filename}.tsv")

    with open(complete_output_path, 'a', encoding='utf-8') as complete_output, \
            open(channel_output_path, 'a', encoding='utf-8') as channel_output:
        for line in output_lines:
            complete_output.write(line + '\n')
            channel_output.write(line + '\n')

    sorted_temp_dis = dsc.sort_dict(temp_auth_dis)
    with open(os.path.join(target_dir, f"{base_filename}_audis.tsv"), 'a', encoding='utf-8') as channel_audis:
        channel_audis.write(dsc.output_string(sorted_temp_dis))

    active_temp_dis = [entry for entry in sorted_temp_dis if sorted_temp_dis[entry] > 99]
    with open(os.path.join(target_dir, f"{base_filename}_active_audis.tsv"), 'a', encoding='utf-8') as active_audis:
        active_audis.write(dsc.output_string(active_temp_dis))


def save_final_distributions():
    sorted_au_dis = dsc.sort_dict(complete_auth_dis)
    with open(os.path.join(target_dir, 'auth_dis.tsv'), 'a', encoding='utf-8') as ad_file:
        ad_file.write(dsc.output_string(sorted_au_dis))

    active_au_dis = [entry for entry in sorted_au_dis if sorted_au_dis[entry] > 99]
    with open(os.path.join(target_dir, 'active_dis.tsv'), 'a', encoding='utf-8') as active_file:
        active_file.write(dsc.output_string(active_au_dis))

    with open(os.path.join(target_dir, 'np_pseudodict.txt'), 'w', encoding='utf-8') as pseudodict_file:
        pseudodict_file.write(str(pseudonym_dict))


def main():
    with open(os.path.join(target_dir, 'n-p_preprocessed_min_len.tsv'), 'a', encoding='utf-8') as complete_output:
        complete_output.write('Comment\tAuthor\n')

    for filename in os.listdir(source_dir):
        if filename.endswith('.bak'):
            process_file(filename)

    save_final_distributions()


if __name__ == "__main__":
    main()
