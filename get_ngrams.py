"""Summarize a corpus as ngram counts.
"""

import json

import pandas
from sklearn.feature_extraction.text import CountVectorizer


with open('stopwords.txt') as f:
    STOPWORDS = f.read().split()


def get_ngrams(text, max_n=1):
    """Get counts of ngrams in a text.
    
    argument text: str text
    argument max_n: int maximum number of words in ngrams
    
    returns: pandas.DataFrame with columns 'term' and 'count'
    """
    
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        token_pattern=r'\b[A-Za-z]{2,}\b',
        ngram_range=(1, max_n)
    )
    
    term_matrix = vectorizer.fit_transform([text]).transpose()
    df = pandas.DataFrame.sparse.from_spmatrix(term_matrix, columns=['count'])
    df['count'] = df['count'].sparse.to_dense()
    df['term'] = vectorizer.get_feature_names_out()
    
    return df


if __name__ == '__main__':
    
    with open('literotica_corpus.json', encoding='utf-8') as f:
        lit_corpus = '\n\n'.join(x['text'] for x in json.load(f))
    
    lit_counts = get_ngrams(lit_corpus)
    lit_counts.to_csv('literotica_counts.csv', index=False)
