import pickle
from pprint import pprint


# def load_I2L():
#     with open('ud_features_I2L.pickle', 'rb') as f:
#         upos_to_features = pickle.load(f)
#         pprint(upos_to_features)
#         return upos_to_features
#
# load_I2L()

properties_POS = {'ADJ': ['Degree', 'Number', 'Gender', 'Case'],
 'ADP': ['Number', 'Gender', 'Case'],
 'ADV': ['Degree', 'Abbr'],
 'AUX': ['Mood',
         'Aspect',
         'Tense',
         'Number',
         'Person',
         'VerbForm',
         'Voice'],
 'CCONJ': [],
 'DET': ['Number', 'Gender', 'PronType', 'Definite', 'Case'],
 'NOUN': ['Number', 'Gender', 'Abbr', 'Case'],
 'NUM': ['NumType', 'Number', 'Gender', 'Case'],
 'PART': [],
 'PRON': ['Number', 'Gender', 'Person', 'Poss', 'PronType', 'Case'],
 'PROPN': ['Number', 'Gender', 'Case'],
 'PUNCT': [],
 'SCONJ': [],
 'SYM': [],
 'VERB': ['Mood',
          'Aspect',
          'Tense',
          'Number',
          'Gender',
          'Person',
          'VerbForm',
          'Voice',
          'Case'],
 'X': ['Foreign'],
 '_': []}


I2L_POS = {'Abbr': ['_', 'Yes'],
 'Aspect': ['Perf', '_', 'Imp'],
 'Case': ['Dat', '_', 'Acc', 'Gen', 'Nom', 'Voc'],
 'Definite': ['Ind', 'Def', '_'],
 'Degree': ['Cmp', 'Sup', '_'],
 'Foreign': ['_', 'Yes'],
 'Gender': ['Fem', 'Masc', '_', 'Neut'],
 'Mood': ['Ind', '_', 'Imp'],
 'NumType': ['Mult', 'Card', '_', 'Ord', 'Sets'],
 'Number': ['Plur', '_', 'Sing'],
 'Person': ['3', '1', '_', '2'],
 'Poss': ['_', 'Yes'],
 'PronType': ['Ind', 'Art', '_', 'Rel', 'Dem', 'Prs', 'Ind,Rel', 'Int'],
 'Tense': ['Pres', 'Past', '_'],
 'VerbForm': ['Part', 'Conv', '_', 'Inf', 'Fin'],
 'Voice': ['Pass', 'Act', '_'],
 'upos': ['X',
          'PROPN',
          'PRON',
          'ADJ',
          'AUX',
          'PART',
          'ADV',
          '_',
          'DET',
          'SYM',
          'NUM',
          'CCONJ',
          'PUNCT',
          'NOUN',
          'SCONJ',
          'ADP',
          'VERB']}
