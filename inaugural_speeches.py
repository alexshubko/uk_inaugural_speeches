import os
import re
import tokenize_uk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

with open('./stopwords.txt', 'r') as file:
    stopwords = [line.rstrip() for line in file.readlines()]

speech_files = os.listdir('./resources/')
speech_titles = [title.rstrip('.txt') for title in speech_files]
speech_texts = []


def my_tokenizer(s):
    return [token for token in tokenize_uk.tokenize_words(s)
            if re.search(r'[а-яА-ЯІіЇї]', token) and token not in stopwords]


for speech_file in speech_files:
    with open(f'./resources/{speech_file}', 'r') as f:
        speech_texts.append(f.read().lower())


vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
vector = vectorizer.fit_transform(speech_texts)
vector_to_array = vector.toarray()

# with open("./top10-tfidf.txt", "w") as output:
#
#     for index, words in enumerate(vector_to_array):
#         output.write(f'top 10 words in \'{speech_titles[index]}\':\n')
#         output.write(str([vectorizer.get_feature_names()[i] for i in list(words.argsort()[-10:])])+"\n\n")

tfidf_df = pd.DataFrame(vector_to_array, speech_titles, columns=vectorizer.get_feature_names())
tfidf_df = tfidf_df.stack().reset_index()
tfidf_df = tfidf_df.rename(columns={0: 'tfidf', 'level_0': 'inaugural_speech', 'level_1': 'term'})
tfidf_df = tfidf_df.sort_values(by=['inaugural_speech','tfidf'], ascending=[True,False]).groupby(['inaugural_speech']).head(10)

with open("./top10-tfidf.csv", "w") as output:
    output.write(tfidf_df.to_csv(index=False))
