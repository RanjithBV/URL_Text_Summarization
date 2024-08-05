import os
import re
import pandas as pd
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import string

positive_words_dir = 'D:/July 2024/BlackCoffee/20211030 Test Assignment/positive_negative_words/positive-words.txt'
negative_words_dir = 'D:/July 2024/BlackCoffee/20211030 Test Assignment/positive_negative_words/negative-words.txt'
stop_words_dir = 'D:/July 2024/BlackCoffee/20211030 Test Assignment/stop_words/stop-words.txt'


def load_words(file_path):
    if os.path.exists(file_path) and os.access(file_path, os.R_OK):
        with open(file_path, 'r') as file:
            words = set(file.read().split())
        return words
    else:
        print(f"Error: Cannot read file {file_path}. Please check the file path and permissions.")
        return set()


positive_words = load_words(positive_words_dir)
negative_words = load_words(negative_words_dir)
stop_words = load_words(stop_words_dir)

d = cmudict.dict()


def count_syllables(word):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0] if word.lower() in d else 1


def compute_variables(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    words = [word.lower() for word in words if word.isalpha()]

    words = [word for word in words if word not in stop_words]

    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)

    avg_sentence_length = len(words) / len(sentences)

    complex_words = [word for word in words if count_syllables(word) > 2]
    percentage_complex_words = len(complex_words) / len(words)

    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    avg_words_per_sentence = len(words) / len(sentences)

    complex_word_count = len(complex_words)

    word_count = len(words)

    syllables_per_word = sum(count_syllables(word) for word in words) / len(words)

    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))

    avg_word_length = sum(len(word) for word in words) / len(words)

    return {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "polarity_score": polarity_score,
        "subjectivity_score": subjectivity_score,
        "avg_sentence_length": avg_sentence_length,
        "percentage_complex_words": percentage_complex_words,
        "fog_index": fog_index,
        "avg_words_per_sentence": avg_words_per_sentence,
        "complex_word_count": complex_word_count,
        "word_count": word_count,
        "syllables_per_word": syllables_per_word,
        "personal_pronouns": personal_pronouns,
        "avg_word_length": avg_word_length
    }


input_dir = 'D:/July 2024/BlackCoffee/20211030 Test Assignment/articles'

input_file_path = 'D:/July 2024/BlackCoffee/20211030 Test Assignment/Input.xlsx'
input_df = pd.read_excel(input_file_path)

output_df = input_df.copy()

for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    article_file = os.path.join(input_dir, f"{url_id}.txt")

    if os.path.exists(article_file) and os.access(article_file, os.R_OK):
        with open(article_file, 'r', encoding='utf-8') as file:
            text = file.read()

        variables = compute_variables(text)

        for key, value in variables.items():
            output_df.at[index, key.upper()] = value
    else:
        print(f"Error: Cannot read file {article_file}. Please check the file path and permissions.")

output_df.to_csv('D:/July 2024/BlackCoffee/20211030 Test Assignment/Output.csv', index=False)
print("Text analysis complete. Output saved to Output.csv.")
