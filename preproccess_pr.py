# -*- coding: utf-8 -*-
import jieba
import openpyxl
from collections import Counter
import gensim
from numpy.random import rand
import re

#stop words
stpwrdpath = "data/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrdlst = [wrd.decode('GBK', 'ignore') for wrd in stpwrdlst]
stpwrd_dic.close()

#emoji
emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"U+2605-U+2b52"                 
        u"U+66d"           "]+", flags=re.UNICODE)
#digits & english & punctuation
dig_eng = re.compile("[\d\w]*")
punc1 = [".","_","(",")","/","-","~",'"',"'","#","...","!","....","*","&"]
punc2 = [u"【",u"】",u"（",u"）",u"…",u"—",u"·",u"～",
         u"#",u"／",u"★",u"❤️️️️️️️",u"」",u"「",u"→",u"＂",u"“",u"ಥ",u"❤"]
punc3 = stpwrdlst[:13]
#vocabulary
model = gensim.models.KeyedVectors.load_word2vec_format('data/cn.skipgram.bin',
                                                        binary=True, unicode_errors='ignore')
vocab = model.vocab

mydic = "data/dict.txt"
jieba.load_userdict(mydic)

movies = ["Thor: Ragnarok","Justice League","Never Say Die"]
dir = "data/predict/"
files = [dir+mov for mov in movies]

results = [[] for i in range(len(files))]
for i,file in enumerate(files):
    with open(file, 'r') as f:
        data = f.read()
        splat = data.split("\n\n")
        for para in splat:
            document_decode = para
            document_cut = jieba.cut(document_decode)
            # remove emojis
            document_cut = ' '.join(document_cut)
            document_cut = emoji.sub(r'', document_cut)
            # remove digits
            document_cut = dig_eng.sub(r'', document_cut)
            # remove double spaces
            document_cut = document_cut.replace('  ', ' ')
            # remove duplicate revies
            document_cut = document_cut.split(' ')
            # remove '' and ' '
            document_cut = [wrd for wrd in document_cut if wrd != '' and wrd != ' ']
            # remove punctuations
            document_cut = [wrd for wrd in document_cut if wrd not in punc1
                            and wrd not in punc2 and wrd not in punc3]
            # remove too short reviews
            if len(document_cut) < 4:
                continue
            result = ' '.join(document_cut)
            result = result.encode('utf-8')
            results[i].append(result)

for i,file in enumerate(files):
    with open(file+'1', 'w+') as f2:
        for res in results[i]:
            f2.write(res+'\n')
            f2.write('\n')
            f2.flush()

