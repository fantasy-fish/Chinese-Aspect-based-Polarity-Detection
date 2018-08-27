#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter

#stop words
stpwrdpath = "data/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrdlst = [wrd.decode('GBK', 'ignore') for wrd in stpwrdlst]
stpwrd_dic.close()

#movie list
movie = "Never Say Die"
dir = "data/predict/"

text_path = dir+movie+"1"
pred_path = dir+movie+"2"

words_plot = [[] for _ in range(3)]
words_cast = [[] for _ in range(3)]
#for counting numbers
list_plot = []
list_cast = []

text = open(text_path).read()
text = text.split("\n\n")
pred = open(pred_path).readlines()

for i in range(len(pred)):
    p = pred[i].split(' ')
    text_index = int(p[0])
    para = text[text_index]
    words = para.split(' ')
    #words = [unicode(wrd,"utf-8").encode('utf-8') for wrd in words]
    words = [wrd.decode('UTF-8') for wrd in words if wrd.decode('UTF-8') not in stpwrdlst]
    plot_polarity = int(p[1])
    cast_polarity = int(p[2][0])
    list_plot.append(plot_polarity)
    list_cast.append(cast_polarity)
    words_plot[plot_polarity] += words
    words_cast[cast_polarity] += words

counter_plot = Counter(list_plot)
percent_plot = [(i, 1.0*counter_plot[i] / len(list_plot)) for i in counter_plot]
print "Percentage of each polarity towards plot:"
print percent_plot
counter_cast = Counter(list_cast)
percent_cast = [(i, 1.0*counter_cast[i] / len(list_cast)) for i in counter_cast]
print "Percentage of polarity towards cast:"
print percent_cast

with open("results/plot_"+movie, 'w+') as f:
    f.write( "Most common words for each polarity towards plot:\n")
    pcounter_plot = [Counter(words) for words in words_plot]
    for c in pcounter_plot:
        for wrd, _ in c.most_common(50):
            f.write(wrd.encode('utf-8')+',')
        f.write('\n\n')
    f.write('---------------------------------------\n')
    #write the reviews
    f.write("All the reviews:\n")
    for i in range(len(pred)):
        p = pred[i].split(' ')
        text_index = int(p[0])
        plot_polarity = int(p[1])
        para = text[text_index]
        f.write(para+'\n')
        f.write(str(plot_polarity)+'\n')


with open("results/cast_"+movie, 'w+') as f:
    f.write("Most common words for each polarity towards cast:\n")
    pcounter_cast = [Counter(words) for words in words_cast]
    for c in pcounter_cast:
        for wrd, _ in c.most_common(100):
            f.write(wrd.encode('utf-8')+',')
        f.write('\n\n')
    f.write('---------------------------------------\n')
    #write the reviews
    f.write("All the reviews:\n")
    for i in range(len(pred)):
        p = pred[i].split(' ')
        text_index = int(p[0])
        cast_polarity = int(p[2][0])
        para = text[text_index]
        f.write(para+'\n')
        f.write(str(cast_polarity)+'\n')
