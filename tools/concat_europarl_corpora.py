#!/usr/bin/env python
# -*- coding: utf-8 -*-

__doc__ = '''
Sentence Aligner for Europarl corpora
'''

import pandas as pd
import subprocess


if __name__ == '__main__':

#    lang_pairs = ['de-en', 'es-en', 'fr-en']
    lang_pairs = ['bg-en', 'cs-en', 'da-en', 'de-en', 'el-en',
                  'es-en', 'et-en', 'fi-en', 'fr-en', 'hu-en',
                  'it-en', 'lt-en', 'lv-en', 'nl-en', 'pl-en',
                  'pt-en', 'ro-en', 'sk-en', 'sl-en', 'sv-en']


    # Load & preprocess CSV files of parallel corpora
    lang_list = list(set('-'.join(lang_pairs).split('-')))
    df_list = []

    for lang_pair in lang_pairs:
        print(lang_pair)

        lang1, lang2 = lang_pair.split('-')

        path_corpus_lang1 = './../corpora/europarl/europarl-v7.{lang_pair}.{lang_corpus}'.format(lang_pair=lang_pair, lang_corpus=lang1)
        path_corpus_lang2 = './../corpora/europarl/europarl-v7.{lang_pair}.{lang_corpus}'.format(lang_pair=lang_pair, lang_corpus=lang2)

        df_lang1 = pd.read_csv(path_corpus_lang1, sep='<<<', engine='python', skip_blank_lines=False)
        df_lang2 = pd.read_csv(path_corpus_lang2, sep='<<<', engine='python', skip_blank_lines=False)

        df = pd.merge(left=df_lang1, right=df_lang2, left_index=True, right_index=True)
        df.columns = [lang1, lang2]

        df = df.drop_duplicates('en')
        df = df.loc[df.iloc[:, 0].notnull() & df.iloc[:, 1].notnull(), :]  # Remove NaN

        print(df.shape)
        print(df.head())

        df_list.append(df)

        
    # Join parallel corpora
    df_merge = df_list[0]

    for i, _df in enumerate(df_list[1:]):
        df_merge = pd.merge(left=df_merge, right=_df, on='en', how='outer')


    # Output as text files
    for lang in lang_list:
        path_output = './../corpora/europarl/europarl-v7.merged.{lang}'.format(lang=lang)
        print(path_output)

        with open(path_output, 'w') as f:
            f.write('\n'.join(df_merge.ix[:, lang].fillna('').tolist()) + '\n')


    # Tokenize all text files
    p = subprocess.call('bash preprocess_merged_corpora.sh', shell=True)
