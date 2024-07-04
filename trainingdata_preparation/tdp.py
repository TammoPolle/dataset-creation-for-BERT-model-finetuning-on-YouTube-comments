import os
from dataset_creation import dsc
from transformers import BertTokenizer, AutoTokenizer


#########
# Helper functions for trainingdata_preparation
#########

# with a list of active authors, write a file with exclusively their comments from a source dataset
# @returns the path of the active subset
def create_active_subset(author_path: str, source_path: str, target_path: str, amount_of_authors: int,
                         comments_per_author: int):
    per_author = {}
    with open(author_path, 'r', encoding='utf-8') as author_file:
        authors = [author.strip() for author in author_file.readlines()[:amount_of_authors]]
        per_author = {author: 0 for author in authors}

    target_file_path = os.path.join(target_path, 'active_set.tsv')
    with open(target_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("Comment\tAuthor\n")

        with open(source_path, 'r', encoding='utf-8') as dataset_file:
            for line in dataset_file:
                temp_author = dsc.get_author(line)
                if temp_author in per_author and per_author[temp_author] < comments_per_author:
                    output_file.write(line)
                    per_author[temp_author] += 1
    return target_file_path


# with the list of authors, write file with comments listed by author from source dataset and write a file with
# their comments for each author
def sort_author_wise(author_path: str, source_path: str, target_path: str):
    with open(source_path, "r", encoding="utf-8") as input_file:
        comments = input_file.readlines()[1:]
    with open(author_path, "r", encoding="utf-8") as author_file:
        authors = [author.strip() for author in author_file.readlines()]

    author_files = {author: open(os.path.join(target_path, f"{author}.tsv"), "w", encoding="utf-8") for author in
                    authors}
    with open(os.path.join(target_path, "author_wise.tsv"), "w", encoding="utf-8") as output_file:
        for comment in comments:
            temp_author = comment.split("\t")[1].strip()
            if temp_author in author_files:
                output_file.write(comment)
                author_files[temp_author].write(comment)

    for file in author_files.values():
        file.close()


# creates a version without authors from source dataset
# @returns the path of the new file
def strip_authors(source_path: str):
    target_path = source_path.replace('.tsv', '_authorfree.tsv')
    with open(source_path, 'r', encoding='utf8') as input_file:
        with open(target_path, 'w', encoding='utf8') as output_file:
            for line in input_file:
                comment = line.split('\t')[0]
                output_file.write(comment + '\n')
    return target_path


# creates new file with all the comments which are fitting into the specified model
# @returns path of the new file
def force_maxlength(model: str, max_length: int, source_path: str):
    if model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model == 'bertweet-base':
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    else:
        raise AttributeError('Error interpreting model name. Expecting "bert-base" or "bertweet-base"')

    target_path = source_path.replace('.tsv', '_short.tsv')
    with open(source_path, "r", encoding="utf-8") as source_file:
        with open(target_path, "w", encoding="utf-8") as output_file:
            for line in source_file:
                if len(tokenizer.tokenize(line.split('\t')[0])) < max_length:
                    output_file.write(line)
    return target_path


# create a new file with the wished amount of authors labeled, every comment from additional authors will be labeled
# with "0" (dump-class)
# @returns the path of the new file
def relabel(source_path: str, label_amount: int):
    with open(source_path, "r", encoding="utf-8") as source_file:
        comments = source_file.readlines()[1:]

    authors = [comment.split('\t')[1].strip() for comment in comments]
    author_list = list(set(authors))
    author_to_label = {author: (i + 1 if i < label_amount else 0) for i, author in enumerate(author_list)}

    target_path = source_path.replace('.tsv', '_labeled.tsv')
    with open(target_path, 'w', encoding="utf-8") as output_file:
        for comment in comments:
            author = comment.split('\t')[1].strip()
            label = author_to_label[author]
            comment = comment.replace(author, str(label))
            output_file.write(comment)
    return target_path


def create_authorship_distribution(target_path: str):
    output_string = "==========\nExperimental Data Authorship Distribution\n==========\n"
    filenames = [
        "active_set.tsv", "active_set_shot.tsv", "limited_labeled.tsv",
        "A_training_set.csv", "A_testing_val_set.csv", "training_set.csv", "testing_val_set.csv"
    ]

    for filename in filenames:
        file_path = os.path.join(target_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()[1:]
                temp_ad = {}
                for line in lines:
                    temp_author = line.split('\t')[1].strip()
                    temp_ad[temp_author] = temp_ad.get(temp_author, 0) + 1
                temp_ad = dsc.sort_dict(temp_ad)
                total_comments = sum(temp_ad.values())
                output_string += f"\n{filename}:\n{total_comments} comments\n{dsc.output_string(temp_ad)}"

    with open(os.path.join(target_path, "authorship_distribution.txt"), "w", encoding="utf-8") as output_file:
        output_file.write(output_string)


# Experiment evaluation: count the correct classifications, the mistakes and the distribution per author
# @returns the string to write in comparative_date file
def evaluation(validation_path: str, prediction_path: str, number_of_labels: int):
    with open(validation_path, 'r', encoding='utf-8') as val_file, open(prediction_path, 'r',
                                                                        encoding='utf-8') as pred_file:
        validation_set = val_file.readlines()
        predictions = pred_file.readlines()

    in_total = len(predictions)
    correctly_classified = 0
    correct_dis = {}
    incorrect_dis = {}

    for val, pred in zip(validation_set, predictions):
        true_class = val.split('\t')[1].strip()
        pred_class = pred.split('\t')[1].strip()
        if true_class == pred_class:
            correctly_classified += 1
            correct_dis[true_class] = correct_dis.get(true_class, 0) + 1
        else:
            incorrect_dis[true_class] = incorrect_dis.get(true_class, 0) + 1

    stats = []
    for i in range(number_of_labels):
        label = str(i)
        correct = correct_dis.get(label, 0)
        incorrect = incorrect_dis.get(label, 0)
        total = correct + incorrect
        accuracy = round(correct / total, 2) if total > 0 else 0
        stats.append(f"{label}:{total}:{correct}:{incorrect}:{accuracy}")

    output_string = (f"Evaluation:\n Number of predictions: {in_total}\t Correctly classified: {correctly_classified}"
                     f"\t Misclassified: {in_total - correctly_classified}\n Accuracy: {correctly_classified / in_total}"
                     f"\n per Author:\nlabel\ttotal\tcorrect\tincorrect\taccuracy\n{dsc.output_string(stats)}")
    return output_string


def limit_comments(source_file: str, target_path: str, comments_per_author: int):
    per_author = {}
    target_file_path = os.path.join(target_path, 'limited_labeled.tsv')
    with open(source_file, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(target_file_path, 'w', encoding='utf-8') as target:
        for line in lines:
            temp_author = dsc.get_author(line)
            if per_author.get(temp_author, 0) < comments_per_author:
                target.write(line)
                per_author[temp_author] = per_author.get(temp_author, 0) + 1
    return target_file_path
