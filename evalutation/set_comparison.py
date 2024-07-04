import pandas as pd

# Dateipfade zu den beiden CSV-Dateien
pathA = "E:\\research\\comment analysis\\n-p\\"
pathB = "E:\\research\\comment analysis\\gen\\"

# overall data fileA (hardcoded)
avg_length_A = 137.737
var_length_A = 0.6506
punct_ratio_A = 0.0273
emo_ratio_A = 0.0016

# overall data fileB (hardcoded)
avg_length_B = 124.5151
var_length_B = 0.6713
punct_ratio_B = 0.025
emo_ratio_B = 0.0009

# Berechnung der Differenzen
length_diff = avg_length_A - avg_length_B
var_diff = var_length_A - var_length_B
punct_diff = punct_ratio_A - punct_ratio_B
emo_diff = emo_ratio_A - emo_ratio_B

# get word_lists
word_list_A = pd.read_csv(pathA + "overall_words.csv", dtype=str)
word_list_B = pd.read_csv(pathB + "overall_words.csv", dtype=str)

# Wörter finden, die nicht in beiden Datensätzen gelistet sind
unique_words_A = word_list_A[~word_list_A['word'].isin(word_list_B['word'])]
unique_words_B = word_list_B[~word_list_B['word'].isin(word_list_A['word'])]

# Sortieren nach Häufigkeit und Begrenzung auf die Top-Wörter
unique_words_A = unique_words_A.sort_values(by='frequency', ascending=False).head(100)
unique_words_B = unique_words_B.sort_values(by='frequency', ascending=False).head(100)

# Top 1000 Wörter der beiden Datensätze
top_1000_words_A = word_list_A.sort_values(by='frequency', ascending=False).head(1000)
top_1000_words_B = word_list_B.sort_values(by='frequency', ascending=False).head(1000)

# Wörter finden, die nicht in den Top 1000 des anderen Datensatzes sind
exclusive_top_words_A = top_1000_words_A[~top_1000_words_A['word'].isin(top_1000_words_B['word'])].head(100)
exclusive_top_words_B = top_1000_words_B[~top_1000_words_B['word'].isin(top_1000_words_A['word'])].head(100)

# Ausgabe in eine Textdatei schreiben
output_file = 'E:\\research\\comment analysis\\comparison_results.txt'
with open(output_file, 'w') as file:
    file.write("Comparison of Overall Data\n")
    file.write("==========================\n\n")

    file.write("Overall Average Length (n-p): {}\n".format(avg_length_A))
    file.write("Overall Average Length (gen): {}\n".format(avg_length_B))
    file.write("Difference in Overall Average Length: {}\n\n".format(length_diff))

    file.write("Variance per Comment (n-p): {}\n".format(var_length_A))
    file.write("Variance per Comment (gen): {}\n".format(var_length_B))
    file.write("Difference in Variance per Comment: {}\n\n".format(var_diff))

    file.write("Punctuation Ratio (n-p): {}\n".format(punct_ratio_A))
    file.write("Punctuation Ratio (gen): {}\n".format(punct_ratio_B))
    file.write("Difference in Punctuation Ratio: {}\n\n".format(punct_diff))

    file.write("Emoji Ratio (n-p): {}\n".format(emo_ratio_A))
    file.write("Emoji Ratio (gen): {}\n".format(emo_ratio_B))
    file.write("Difference in Emoji Ratio: {}\n\n".format(emo_diff))

    file.write("Unique Words in n-p\n")
    file.write("=======================\n")
    file.write(unique_words_A.to_string(index=False))
    file.write("\n\n")

    file.write("Unique Words in gen\n")
    file.write("=======================\n")
    file.write(unique_words_B.to_string(index=False))
    file.write("\n\n")

    file.write("Exclusive Top Words in n-p (not in top 1000 of gen)\n")
    file.write("=========================================\n")
    file.write(exclusive_top_words_A.to_string(index=False))
    file.write("\n\n")

    file.write("Exclusive Top Words in gen (not in top 1000 of n-p)\n")
    file.write("=========================================\n")
    file.write(exclusive_top_words_B.to_string(index=False))

print("Comparison results saved to {}".format(output_file))
