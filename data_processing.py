'''
    清洗数据,转换语料格式,得到字嵌入
    author:xierhacker
    time:2018.1.4
'''
import re
import os
import time
import pandas as pd
import numpy as np
from itertools import chain
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from parameter import MAX_SENTENCE_SIZE

#原始语料转换为不带任何标记的语料,可以训练字向量
def toCharCorpus(filename):
    doc = ""
    file = open(file=filename, encoding="utf-8")
    lines = file.readlines()
    # 匹配#标记
    pattern1 = re.compile(r"#[1,2,3,4]", flags=re.U)
    # 每个字匹配一次
    pattern2 =re.compile(r"[^\s]")
    for line in lines:
        string = re.sub(pattern=pattern1, repl="", string=line)     #去掉#
        string=" ".join(re.findall(pattern=pattern2,string=string)) #每个字加上空格
        string+="\n"
        doc += string
    # write to file
    f = open(file="./data/corpus/prosody_char.txt", mode="w", encoding="utf-8")
    f.write(doc)
    f.close()

#训练字向量并且存储
def toEmbeddings(filename):
    sentences = word2vec.Text8Corpus(filename)
    model = word2vec.Word2Vec(sentences=sentences, size=1001)
    # save embeddings file
    if not os.path.exists("./data/embeddings"):
        os.mkdir(path="./data/embeddings")
    model.wv.save_word2vec_format("./data/embeddings/vec.txt", binary=False)


#转换原始corpus为韵律词(PW)格式标记
def toPW(filename):
    doc=""
    file = open(file=filename, encoding="utf-8")
    lines = file.readlines()
    # 匹配#标记
    pattern1 = re.compile(r"#[1,2,3,4]", flags=re.U)
    # 不是/或者b
    pattern2 = re.compile(r"(?![/b])")
    #去掉b后面的/n
    pattern3= re.compile(r"b/n")
    #去掉开头的/n
    pattern4=re.compile(r"^/n")
    for line in lines:
        line=line.strip()
        string = re.sub(pattern=pattern1, repl="/b", string=line)  # 去掉#
        string=re.sub(pattern=pattern2,repl="/n",string=string)
        string = re.sub(pattern=pattern3, repl="b", string=string)
        string = re.sub(pattern=pattern4, repl="", string=string)+"\n"
        doc += string
    # write to file
    f = open(file="./data/corpus/prosody_pw.txt", mode="w", encoding="utf-8")
    f.write(doc)
    f.close()

#转换原始corpus为韵律短语(PPH)格式标记
def toPPH(filename):
    doc=""
    file = open(file=filename, encoding="utf-8")
    lines = file.readlines()
    #匹配#1(因为要先去掉#1)
    pattern=re.compile(r"#1")
    # 匹配#标记
    pattern1 = re.compile(r"#[2,3,4]", flags=re.U)
    # 不是/或者b
    pattern2 = re.compile(r"(?![/b])")
    #去掉b后面的/n
    pattern3= re.compile(r"b/n")
    #去掉开头的/n
    pattern4=re.compile(r"^/n")
    for line in lines:
        line=line.strip()   #去掉一些影响的空格和换行
        string = re.sub(pattern=pattern, repl="", string=line)  # 去掉#1
        string = re.sub(pattern=pattern1, repl="/b", string=string)  # 去掉#
        string=re.sub(pattern=pattern2,repl="/n",string=string)
        string = re.sub(pattern=pattern3, repl="b", string=string)
        string = re.sub(pattern=pattern4, repl="", string=string)+"\n"
        doc += string
    # write to file
    f = open(file="./data/corpus/prosody_pph.txt", mode="w", encoding="utf-8")
    f.write(doc)
    f.close()

#转换原始corpus为语调短语(IPH)格式标记
def toIPH(filename):
    doc = ""
    file = open(file=filename, encoding="utf-8")
    lines = file.readlines()
    # 匹配#1和#2(因为要先去掉#1和#2)
    pattern = re.compile(r"#[1,2]")
    # 匹配#标记
    pattern1 = re.compile(r"#[3,4]", flags=re.U)
    # 不是/或者b
    pattern2 = re.compile(r"(?![/b])")
    # 去掉b后面的/n
    pattern3 = re.compile(r"b/n")
    # 去掉开头的/n
    pattern4 = re.compile(r"^/n")
    for line in lines:
        line = line.strip()  # 去掉一些影响的空格和换行
        string = re.sub(pattern=pattern, repl="", string=line)  # 去掉#1
        string = re.sub(pattern=pattern1, repl="/b", string=string)  # 去掉#
        string = re.sub(pattern=pattern2, repl="/n", string=string)
        string = re.sub(pattern=pattern3, repl="b", string=string)
        string = re.sub(pattern=pattern4, repl="", string=string) + "\n"
        doc += string
    # write to file
    f = open(file="./data/corpus/prosody_iph.txt", mode="w+", encoding="utf-8")
    f.write(doc)
    f.close()

#清洗
def clean(s):
    if u'“/s' not in s:                 # 句子中间的引号不应去掉
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

def file2corpus(filename):
    '''
    :param filename:
    :return: 语料文件文件转换为一个原始语料句子的list
    '''
    with open(filename, 'rb') as inp:
        corpus = inp.read().decode('UTF-8')   #原始语料 str对象
    corpus = corpus.split('\r')           #换行切分,得到一个简陋列表
    corpus = u''.join(map(clean, corpus))   # 把所有处理的句子连接起来,这里中间连接不用其他字符 str对象
    corpus = re.split(u"\n", corpus)  # 以换行为分割,把语料划分为一个"句子"列表
    #corpus = re.split(u'[，。！？、‘’“”]/[bems]', corpus)    # 以换行为分割,把语料划分为一个"句子"列表
    return corpus              #[人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e ,xxx,....]


def make_component(corpus,name):
    '''
    :param corpus: 传入原始语料句子corpus列表得到的字数据datas和对应的labels数据都放到dataframe里面存储,方便后面的处理
    :return: df_data
    '''
    sentences= []
    tags = []
    for s in corpus:           #corpus列表得到每句corpus想应的sentence以及对应的labels
        sentence_tags = re.findall('(.)/(.)', s)     # sentence_tags:[('人', 'b'), ('们', 'e'), ('常', 's'), ('说', 's')]
        if sentence_tags:                            # 顺便去除了一些空样本
            sentence_tags = np.array(sentence_tags)
            sentences.append(sentence_tags[:, 0])    #sentences每一个元素表示一个sentence['人' '们' '常' '说' '生' '活' '是' '一' '部' '教' '科' '书']
            tags.append(sentence_tags[:, 1])         #tags每一个元素表示的是一个句子对应的标签['b' 'e' 's' 's' 'b' 'e' 's' 's' 's' 'b' 'm' 'e']

    #使用pandas处理,简化流程
    df_data = pd.DataFrame({'sentences': sentences, 'tags': tags}, index=range(len(sentences)))
    df_data['sentence_len'] = df_data['sentences'].apply(lambda sentences: len(sentences))  # 每句话长度

    # 得到所有的字,这里的all_words是一个列表,存放了这个语料中所有的词
    all_words = list(chain(*df_data['sentences'].values))
    sr_allwords = pd.Series(data=all_words)     # 2.列表做成pandas的Series
    words = (sr_allwords.value_counts()).index  #字列表.统计每个字出现的频率,同时相当于去重复,得到字的集合(这里还是Serieas的index对象)
    #print(lenwords)

    words_id = range(1, len(words) + 1)         #字的id列表,从1开始，因为准备把0作为填充值
    tags = ['x', 'n', 'b']                      #tag列表
    tags_id = range(len(tags))                  #tag的id列表

    #保存基本组件,在./dataset/name/下面会有words_ids.csv,tags_ids.csv,df_data.csv三个文件
    if not os.path.exists("./dataset/"):
        os.mkdir("./dataset/")
    if not os.path.exists("./dataset/"+name):
        os.mkdir("./dataset/"+name)

    # words以及对应的id组件
    pd.DataFrame(data={"words":words,"id":words_id}).\
        to_csv(path_or_buf="./dataset/"+name+"/words_ids.csv",index=False,encoding="utf_8")
    # tags以及对应的id组件
    pd.DataFrame(data={"tags":tags,"id":tags_id}).\
        to_csv(path_or_buf="./dataset/"+name+"/tags_ids.csv",index=False,encoding="utf_8")
    print("max_sentence_size:",df_data["sentence_len"].max())
    return df_data      #暂时不保存,返回

def read_component(name):
    '''
    从文件里面读取基本的component
    :param name:
    :return: words2id, id2words, tags2id, id2tags
    '''
    #读取words和ids的dataframe
    df_words_ids=pd.read_csv(filepath_or_buffer="./dataset/"+name+"/words_ids.csv",encoding="utf-8")
    #读取tags和ids的dataframe
    df_tags_ids=pd.read_csv(filepath_or_buffer="./dataset/"+name+"/tags_ids.csv",encoding="utf-8")
    #装换为words2id, id2words, tags2id, id2tags
    #df_data=pd.DataFrame(data={})
    words2id=pd.Series(data=df_words_ids["id"].values,index=df_words_ids["words"].values)
    id2words=pd.Series(data=df_words_ids["words"].values,index=df_words_ids["id"].values)
    tags2id = pd.Series(data=df_tags_ids["id"].values, index=df_tags_ids["tags"].values)
    id2tags = pd.Series(data=df_tags_ids["tags"].values, index=df_tags_ids["id"].values)
    return words2id, id2words, tags2id, id2tags


def make_dataset(in_filename,project_name,out_filename):
    '''
    转换为最后模型适合的数据集,name表示转换后的数据集存储在哪个文件下面./dataset/name
    :param filename: 原始数据集的文件名
    :param name:
    :return:
    '''
    print("Conver Corpus To Dataset!")
    start_time=time.time()
    corpus = file2corpus(in_filename); print("         corpus contains ", len(corpus), " sentences.")
    #保存基本组件,并且返回
    print("         ----saving component <tags_ids.csv> and <words_ids.csv>")
    df_data=make_component(corpus,project_name)
    #print(df_data.head(3))

    #读取组件,并且装换为合适的格式
    words2id, id2words, tags2id, id2tags =read_component(project_name)
    print("words2id.shape:",words2id.shape)
    print("         dataset contains ",df_data.shape[0]," sentences.")

    def X_padding(sentence):
        '''
        !!!!!!!!!!可提速!!!!!!!!
        返回一句话padding之后的id列表,使用的时候,把一个字符串转为list传进来就行
        :param sentence: 一个句子的列表
        :param word2id: word2id映射
        :return:   一句话的padding后的 ids

        '''
        ids = list(words2id[sentence])
        if len(ids) > MAX_SENTENCE_SIZE:  # 超过就截断
            return ids[:MAX_SENTENCE_SIZE]
        if len(ids) < MAX_SENTENCE_SIZE:  # 短了就补齐
            ids.extend([0] * (MAX_SENTENCE_SIZE - len(ids)))
        return ids

    def y_padding(tags):
        '''
            !!!!!!!!!!!可提速!!!!!!!!!
        #得到一个label的padding后的id
        :param tags:
        :param tags2id:
        :return:
        '''
        ids = list(tags2id[tags])
        if len(ids) > MAX_SENTENCE_SIZE:  # 超过就截断
            return ids[:MAX_SENTENCE_SIZE]
        if len(ids) < MAX_SENTENCE_SIZE:  # 短了就补齐
            ids.extend([0] * (MAX_SENTENCE_SIZE - len(ids)))
        return ids

    #把数据转换为ids的数据
    print("         convert data and label to 'ids' represented")
    df_data['X'] = df_data['sentences'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    #print(df_data["X"].head(5))
    #print(df_data["y"].head(5))

    #数据集切分0.2 比例
    df_data_train=df_data[:47176]
    df_data_validation=df_data[47176:]
    #df_data_train,df_data_test=train_test_split(df_data,test_size=0.2)              #训练集和测试集
    #df_data_train,df_data_validation=train_test_split(df_data_train,test_size=0.1)  #训练集和验证集

    #保存最终数据到pkl文件
    print("         ----saving final dataset <"+out_filename+"_summary_train.pkl>")
    df_data_train.to_pickle(path="./dataset/"+project_name+"/"+out_filename+"_summary_train.pkl")

    print("         ----saving final dataset <summary_validation.pkl>")
    df_data_validation.to_pickle(path="./dataset/"+project_name+"/"+out_filename+"_summary_validation.pkl")

    duration=time.time()-start_time;
    print("END! this operation spends ",round(duration/60,2)," mins")


#summary_train.pkl

if __name__=="__main__":
    print("[1]-->trans corpus to char corpus and saving......")
    toCharCorpus(filename="./data/corpus/prosody.txt")
    print("[2]-->get embeddings and saving......")
    toEmbeddings(filename="./data/corpus/prosody_char.txt")

    print("[3]-->trans corpus to PW format......")
    toPW("./data/corpus/prosody.txt")
    print("[4]-->trans corpus to PPH format......")
    toPPH("./data/corpus/prosody.txt")
    print("[5]-->trans corpus to IPH format......")
    toIPH("./data/corpus/prosody.txt")

    print("[6]-->trans corpus_pw to dataset......")
    make_dataset(in_filename="./data/corpus/prosody_pw.txt",project_name="temptest",out_filename="pw")

    print("[7]-->trans corpus_pph to dataset......")
    make_dataset(in_filename="./data/corpus/prosody_pph.txt", project_name="temptest", out_filename="pph")

    print("[8]-->trans corpus_iph to dataset......")
    make_dataset(in_filename="./data/corpus/prosody_iph.txt", project_name="temptest", out_filename="iph")








