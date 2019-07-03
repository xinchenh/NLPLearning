import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time
import jieba
import os
from six import iteritems

basedir = "corpus/news/"
dir_list = ['affairs','constellation','economic','edu','ent','fashion','game','home','house','lottery','science','sports','stock']

fw = open("news.tab","w") #保存切分好的文本数据
fw_type = open("type.tab","w") #保存新闻类型，与news.tab一一对应
num = -1
for e in dir_list:
    num += 1
    indir = basedir + e + '/'
    files = os.listdir(indir)
    count = 0
    for file in files:
        if count > 10000: #每个新闻类别取10000篇
            break
        count += 1            
        filepath = indir + file
        with open(filepath,'r') as fr:
            text = fr.read()
        text = text.decode("utf-8").encode("utf-8")
        seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," "))
        outline = " ".join(seg_text) + "\n"
        outline = outline.encode("utf-8")
        #print outline
        fw.write(outline)
        fw.flush()
        fw_type.write(str(num) + "\n")
        fw_type.flush()
fw.close()
fw_type.flush()


def load_stopwords():
    f_stop = open('stopwords.tab')
    sw = [line.strip().decode("utf-8") for line in f_stop]
    f_stop.close()
    return sw
stop_words = load_stopwords()

print '正在建立词典--'
t_start = time.time()
dictionary = corpora.Dictionary(line.split() for line in open('news.tab'))
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()
dictionary.save('corpora.dict') #保存生成的dictionary
print "建立词典完成，用时%.3f秒" % (time.time() - t_start)

#dictionary = corpora.Dictionary.load('corpora.dict') #使用保存的dictionary


print "开始计算文本向量--"
t_start = time.time()
class MyCorpus(object):
    def __iter__(self):
        for line in open("news.tab"):
            yield dictionary.doc2bow(line.split())
corpus_memory_friendly = MyCorpus()
corpus = []
for vector in corpus_memory_friendly:
    corpus.append(vector)
print "计算文本向量完成，用时%.3f秒" % (time.time() - t_start)

print '正在保存文本向量--'
t_start = time.time()
corpora.MmCorpus.serialize("corpus.mm",corpus) #保存生成的corpus向量
print "保存文本向量完成，用时%.3f秒" % (time.time() - t_start)


corpus = corpora.MmCorpus("./corpus.mm") #使用保存的corpus向量
