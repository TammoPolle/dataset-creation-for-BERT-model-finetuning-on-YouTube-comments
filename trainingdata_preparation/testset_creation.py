import pandas as pd
import os
from dataset_creation import dsc


#########
# Helper functions for trainingdata_preparation
########

def create_active_subset(author_path, source_path, target_path, amount_of_authors, comments_per_author):
    per_author = {}
    with open(author_path, 'r', encoding='utf-8') as author_file:
        authors = [author.strip() for author in author_file.readlines()[:amount_of_authors]]
        per_author = {author: 0 for author in authors}

    with open(source_path, 'r', encoding='utf-8') as dataset_file:
        dataset = dataset_file.readlines()

    target_file = os.path.join(target_path, 'active_set.tsv')
    with open(target_file, 'w', encoding='utf-8') as output:
        output.write("Comment\tAuthor\n")
        for line in dataset:
            temp_author = dsc.get_author(line)
            if temp_author in per_author and per_author[temp_author] < comments_per_author:
                output.write(line)
                per_author[temp_author] += 1
    return target_file


def sort_author_wise(author_path, source_path, target_path):
    with open(source_path, "r", encoding="utf-8") as input_file:
        comments = input_file.readlines()[1:]
    with open(author_path, "r", encoding="utf-8") as author_file:
        authors = [author.strip() for author in author_file.readlines()]

    with open(os.path.join(target_path, "author_wise.tsv"), "w", encoding="utf-8") as output_file:
        for author in authors:
            author_comments = [comment for comment in comments if comment.split("\t")[1].strip() == author]
            output_file.writelines(author_comments)
            with open(os.path.join(target_path, f"{author}.tsv"), "w", encoding="utf-8") as author_file:
                author_file.writelines(author_comments)


def strip_authors(source_path):
    target_path = source_path.replace('.tsv', '_authorfree.tsv')
    with open(source_path, 'r', encoding='utf8') as input_file:
        comments_with_author = input_file.readlines()
    with open(target_path, 'w', encoding='utf8') as output_file:
        for comment in comments_with_author:
            author_free_comment = comment.split('\t')[0]
            output_file.write(author_free_comment + '\n')
    return target_path


def force_maxlength(model, max_length, source_path):
    from transformers import BertTokenizer, AutoTokenizer
    if model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model == 'bertweet-base':
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    else:
        raise AttributeError('Error interpreting model name. Expecting "bert-base" or "bertweet-base"')

    target_path = source_path.replace('.tsv', '_short.tsv')
    with open(source_path, "r", encoding="utf-8") as source_file:
        comments = source_file.readlines()
    with open(target_path, "w", encoding="utf-8") as output_file:
        for comment in comments:
            if len(tokenizer.tokenize(comment.split('\t')[0])) < max_length:
                output_file.write(comment)
    return target_path


def relabel(source_path, label_amount):
    with open(source_path, "r", encoding="utf-8") as source_file:
        comments = source_file.readlines()[1:]

    authors = [comment.split('\t')[1].strip() for comment in comments]
    author_list = list(set(authors))
    author_map = {author: i + 1 for i, author in enumerate(author_list[:label_amount])}

    target_path = source_path.replace('.tsv', '_labeled.tsv')
    with open(target_path, 'w', encoding="utf-8") as output_file:
        for comment in comments:
            author = comment.split('\t')[1].strip()
            new_author = str(author_map.get(author, 0))
            labeled_comment = comment.replace(author, new_author)
            output_file.write(labeled_comment)
    return target_path


def create_authorship_distribution(target_path):
    output_string = "==========\nExperimental Data Authorship Distribution\n==========\n"
    for filename in os.listdir(target_path):
        if filename in ["active_set.tsv", "active_set_shot.tsv", "limited_labeled.tsv",
                        "A_training_set.csv", "A_testing_val_set.csv", "training_set.csv", "testing_val_set.csv"]:
            with open(os.path.join(target_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()[1:]
            temp_ad = {}
            for line in lines:
                author = line.split('\t')[1].strip()
                temp_ad[author] = temp_ad.get(author, 0) + 1
            sorted_temp_ad = dsc.sort_dict(temp_ad)
            total_comments = sum(sorted_temp_ad.values())
            output_string += f"\n{filename}:\n{total_comments} comments\n{dsc.output_string(sorted_temp_ad)}"

    with open(os.path.join(target_path, "authorship_distribution.txt"), "w", encoding="utf-8") as output_file:
        output_file.write(output_string)


def evaluation(validation_path, prediction_path, number_of_labels):
    with open(validation_path, 'r', encoding='utf-8') as val_file:
        validation_set = val_file.readlines()
    with open(prediction_path, 'r', encoding='utf-8') as pred_file:
        predictions = pred_file.readlines()

    correct_dis, incorrect_dis = {}, {}
    correctly_classified = 0

    for val, pred in zip(validation_set, predictions):
        true_class = val.split('\t')[1].strip()
        predicted_class = pred.split('\t')[1].strip()
        if true_class == predicted_class:
            correctly_classified += 1
            correct_dis[true_class] = correct_dis.get(true_class, 0) + 1
        else:
            incorrect_dis[true_class] = incorrect_dis.get(true_class, 0) + 1

    total_predictions = len(predictions)
    accuracy = correctly_classified / total_predictions

    stats = []
    for i in range(number_of_labels):
        i = str(i)
        total = correct_dis.get(i, 0) + incorrect_dis.get(i, 0)
        correct = correct_dis.get(i, 0)
        incorrect = incorrect_dis.get(i, 0)
        acc = round(correct / total, 2) if total > 0 else 0
        stats.append(f"{i}:{total}:{correct}:{incorrect}:{acc}")

    output_string = (
        f"Evaluation:\n Number of predictions: {total_predictions}\t Correctly classified: {correctly_classified}"
        f"\t Misclassified: {total_predictions - correctly_classified}\n Accuracy: {accuracy}\n"
        f"per Author:\nlabel\ttotal\tcorrect\tincorrect\taccuracy\n{dsc.output_string(stats)}")
    return output_string


def limit_comments(source_file, target_path, comments_per_author):
    per_author = {}
    target_file = os.path.join(target_path, 'limited_labeled.tsv')
    with open(source_file, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(target_file, 'w', encoding='utf-8') as target:
        for line in lines:
            author = dsc.get_author(line)
            if per_author.get(author, 0) < comments_per_author:
                target.write(line)
                per_author[author] = per_author.get(author, 0) + 1
    return target_file


def main():
    model_name = "bertweet-base"
    source_path = "E:\\research\\datapool\\preprocessed\\combined\\min_len\\"
    target_path = "E:\\research\\experiments\\combined\\Bertweet_50_all\\"
    max_input_length = 128
    label_amount = 50
    comments_per_author = 5000

    active_set_path = create_active_subset(source_path + 'active_dis.tsv', source_path + 'n-p_preprocessed_min_len.tsv',
                                           target_path, label_amount, comments_per_author)
    active_fit_set_path = force_maxlength(model_name, max_input_length, active_set_path)
    dataset_path = relabel(active_fit_set_path, label_amount)
    full_dataset_frame = pd.read_csv(dataset_path, sep='\t', encoding='utf-8')

    training_set = full_dataset_frame.sample(frac=0.8, random_state=42).reset_index(drop=True)
    testing_set = full_dataset_frame.drop(training_set.index).reset_index(drop=True)

    training_set.to_csv(os.path.join(target_path, "training_set.csv"), index=False, sep="\t",
                        header=['Comment', 'Author'])
    testing_set.to_csv(os.path.join(target_path, "testing_val_set.csv"), index=False, sep="\t",
                       header=['Comment', 'Author'])

    with open(os.path.join(target_path, "testing_val_set.csv"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(os.path.join(target_path, "testing_set.csv"), "w", encoding="utf-8") as of:
        of.write("id\tComment\n")
        for i, line in enumerate(lines[1:], start=1):
            of.write(f"{i}\t{line.split('\t')[0]}\n")

    create_authorship_distribution(target_path)


if __name__ == "__main__":
    main()