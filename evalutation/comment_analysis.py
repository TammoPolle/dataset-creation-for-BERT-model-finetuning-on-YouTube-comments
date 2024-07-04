import pandas as pd
import re
from collections import Counter, defaultdict
import nltk


# Einlesen der TSV-Datei
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', names=['Comment', 'Author'])
    df.dropna(subset=['Comment', 'Author'], inplace=True)  # Entfernt Zeilen mit NaN-Werten
    df['Comment'] = df['Comment'].astype(str)  # Stellt sicher, dass alle Kommentare Strings sind
    return df


# Berechnung der durchschnittlichen Zeichenzahl und Varianz
def calculate_lengths(df):
    char_counts = [len(comment) for comment in df['Comment']]
    overall_avg_length = round(sum(char_counts) / len(char_counts), 4)
    overall_var_length = sum((x - overall_avg_length) ** 2 for x in char_counts)
    overall_var_per_comment = round((overall_var_length ** (1 / 2)) / len(char_counts), 4)

    author_stats = {}
    for author in df['Author'].unique():
        author_comments = df[df['Author'] == author]['Comment']
        author_char_counts = [len(comment) for comment in author_comments]
        avg_length = round(sum(author_char_counts) / len(author_char_counts), 4)
        var_length = sum((x - avg_length) ** 2 for x in author_char_counts)
        var_per_comment = round((var_length ** (1 / 2)) / len(author_char_counts), 4)
        author_stats[author] = {
            'avg_length': avg_length,
            'var_per_comment': var_per_comment,
            'comment_count': len(author_comments)
        }

    return overall_avg_length, overall_var_per_comment, author_stats


# Berechnung der Wort- und Satzzeichenhäufigkeit
def calculate_word_and_punctuation_freq(df):
    overall_word_freq = Counter()
    overall_punct_freq = Counter()
    author_word_freq = defaultdict(Counter)
    author_punct_freq = defaultdict(Counter)

    total_word_count = 0

    for _, row in df.iterrows():
        tokens = nltk.word_tokenize(row['Comment'].lower())
        words = [token for token in tokens if token.isalnum()]
        punctuations = [token for token in tokens if re.match(r'^\W+$', token) and not token.isalnum() and token != ':']

        overall_word_freq.update(words)
        overall_punct_freq.update(punctuations)
        author_word_freq[row['Author']].update(words)
        author_punct_freq[row['Author']].update(punctuations)

        total_word_count += len(words)

    return overall_word_freq, overall_punct_freq, author_word_freq, author_punct_freq, total_word_count


# Berechnung der Emoji-Häufigkeit
def calculate_emoji_freq(df):
    overall_emoji_freq = Counter()
    author_emoji_freq = defaultdict(Counter)

    for _, row in df.iterrows():
        emojis = re.findall(r':[^\s:]+:',
                            row['Comment'])  # Emojis sind Zeichenketten zwischen Doppelpunkten ohne Leerzeichen
        overall_emoji_freq.update(emojis)
        author_emoji_freq[row['Author']].update(emojis)

    return overall_emoji_freq, author_emoji_freq


# Berechnung der Verhältnisse
def calculate_ratios(df, char_counts, punct_counts, emoji_counts, total_word_count):
    punctuation_ratios = [round(punct / words, 4) if words > 0 else 0 for punct, words in
                          zip(punct_counts, char_counts)]
    emoji_ratios = [round(emo / comments, 4) if comments > 0 else 0 for emo, comments in zip(emoji_counts, char_counts)]
    overall_punctuation_ratio = round(sum(punctuation_ratios) / len(punctuation_ratios), 4)
    overall_emoji_ratio = round(sum(emoji_ratios) / len(emoji_ratios), 4)

    author_stats = defaultdict(lambda: {'punctuation_ratio': [], 'emoji_ratio': []})
    for author, punct_ratio, emoji_ratio in zip(df['Author'], punctuation_ratios, emoji_ratios):
        author_stats[author]['punctuation_ratio'].append(punct_ratio)
        author_stats[author]['emoji_ratio'].append(emoji_ratio)

    author_ratios = {
        author: {
            'punctuation_ratio': round(sum(stats['punctuation_ratio']) / len(stats['punctuation_ratio']), 4),
            'emoji_ratio': round(sum(stats['emoji_ratio']) / len(stats['emoji_ratio']), 4)
        }
        for author, stats in author_stats.items()
    }

    return overall_punctuation_ratio, overall_emoji_ratio, author_ratios


# Speichern der Ergebnisse in einer CSV-Datei
def save_to_csv(file_path, overall_stats, author_stats, overall_word_freq, overall_punct_freq, author_word_freq,
                author_punct_freq, overall_emoji_freq, author_emoji_freq, overall_ratios, author_ratios,
                total_word_count):
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        # Gesamtstatistiken
        overall_stats_df = pd.DataFrame([overall_stats], columns=['avg_length', 'var_per_comment'])
        overall_stats_df.to_csv(file, index=False)

        overall_ratios_df = pd.DataFrame([overall_ratios], columns=['punctuation_ratio', 'emoji_ratio'])
        overall_ratios_df.to_csv(file, mode='a', index=False)

        overall_word_freq_df = pd.DataFrame(overall_word_freq.items(), columns=['word', 'frequency']).sort_values(
            by='frequency', ascending=False).head(1000)
        overall_word_freq_df['word_ratio'] = (overall_word_freq_df['frequency'] / total_word_count).round(4)
        overall_word_freq_df.to_csv(file, mode='a', index=False)

        overall_punct_freq_df = pd.DataFrame([item for item in overall_punct_freq.items() if item[1] >= 10],
                                             columns=['punctuation', 'frequency']).sort_values(by='frequency',
                                                                                               ascending=False)
        overall_punct_freq_df.to_csv(file, mode='a', index=False)

        overall_emoji_freq_df = pd.DataFrame(overall_emoji_freq.items(), columns=['emoji', 'frequency']).sort_values(
            by='frequency', ascending=False)
        overall_emoji_freq_df.to_csv(file, mode='a', index=False)

        file.write("\n")  # Leerzeile zwischen Overall und Author-Daten

        # Author-Daten
        author_stats_df = pd.DataFrame.from_dict(author_stats, orient='index').reset_index().rename(
            columns={'index': 'Author'})
        author_stats_df = author_stats_df.sort_values(by='comment_count', ascending=False)

        for _, stats in author_stats_df.iterrows():
            file.write(f"Author: {stats['Author']}\n")
            author_info_df = pd.DataFrame([stats], columns=['avg_length', 'var_per_comment', 'comment_count'])
            author_info_df.to_csv(file, mode='a', index=False)

            author_ratios_df = pd.DataFrame(author_ratios[stats['Author']], index=[0])
            author_ratios_df.to_csv(file, mode='a', index=False)

            word_freq_df = pd.DataFrame(author_word_freq[stats['Author']].items(),
                                        columns=['word', 'frequency']).sort_values(by='frequency',
                                                                                   ascending=False).head(100)
            total_author_words = sum(author_word_freq[stats['Author']].values())
            word_freq_df['word_ratio'] = (word_freq_df['frequency'] / total_author_words).round(4)
            word_freq_df.to_csv(file, mode='a', index=False)

            punct_freq_df = pd.DataFrame([item for item in author_punct_freq[stats['Author']].items() if item[1] >= 10],
                                         columns=['punctuation', 'frequency']).sort_values(by='frequency',
                                                                                           ascending=False)
            punct_freq_df.to_csv(file, mode='a', index=False)

            emoji_freq_df = pd.DataFrame(author_emoji_freq[stats['Author']].items(),
                                         columns=['emoji', 'frequency']).sort_values(by='frequency', ascending=False)
            emoji_freq_df.to_csv(file, mode='a', index=False)

            file.write("\n")  # Leerzeile zwischen verschiedenen Author-Daten


# Hauptfunktion
if __name__ == "__main__":
    # Beispiel für das Einlesen der Daten (ändern Sie den Pfad entsprechend)
    df = load_data("E:\\research\\comment analysis\\n-p\\active_set_short.tsv")

    # Berechnung der Längenstatistiken
    overall_avg_length, overall_var_per_comment, author_length_stats = calculate_lengths(df)

    # Berechnung der Wort- und Satzzeichenhäufigkeit
    overall_word_freq, overall_punct_freq, author_word_freq, author_punct_freq, total_word_count = calculate_word_and_punctuation_freq(
        df)

    # Berechnung der Emoji-Häufigkeit
    overall_emoji_freq, author_emoji_freq = calculate_emoji_freq(df)

    # Berechnung der Verhältnisse
    char_counts = [len(comment) for comment in df['Comment']]
    punct_counts = [len([token for token in nltk.word_tokenize(comment.lower()) if
                         re.match(r'^\W+$', token) and not token.isalnum() and token != ':']) for comment in
                    df['Comment']]
    emoji_counts = [len(re.findall(r':[^\s:]+:', comment)) for comment in df['Comment']]
    overall_punctuation_ratio, overall_emoji_ratio, author_ratios = calculate_ratios(df, char_counts, punct_counts,
                                                                                     emoji_counts, total_word_count)

    # Speichern der Ergebnisse in einer CSV-Datei
    save_to_csv('E:\\research\\comment analysis\\n-p\\comment_analysis.csv',
                (overall_avg_length, overall_var_per_comment),
                author_length_stats,
                overall_word_freq,
                overall_punct_freq,
                author_word_freq,
                author_punct_freq,
                overall_emoji_freq,
                author_emoji_freq,
                (overall_punctuation_ratio, overall_emoji_ratio),
                author_ratios,
                total_word_count)

    print('Analysis saved to comment_analysis.csv')
