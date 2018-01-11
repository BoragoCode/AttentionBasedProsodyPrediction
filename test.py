import tensorflow as tf
import numpy as np
import pandas as pd

c=[1,1,1,1,2,1,1,1,0,2]
b=[1,1,1,2,1,1,1,1,0,2]
a=[1,2,1,2,0,1,1,2,0,2]

complex=np.array([c,b,a])
    #np.concatenate((a,b,c),axis=0)
arg=np.argmax(complex,axis=0)
print("arg:\n",arg)
for i in range(10):
    if arg[i]==0:
        if complex[0,i]==2:
            arg[i]=6
        else:
            arg[i]=0
    if arg[i]==1:
        if complex[1,i]==2:
            arg[i]=4
        else:
            arg[i]=0
    if arg[i]==2:
        if complex[2,i]==2:
            arg[i]=2
        else:
            arg[i]=0

    #if arg[i]==2:

arg=(arg/2).astype(dtype=np.int32)
#arg[arg==0]=6
#arg[arg==1]=4
#arg[arg==2]=2
print(complex)
print("arg:\n",arg)
arg=np.reshape(arg,newshape=(5,2))
print(arg)

df_words_ids=pd.read_csv(filepath_or_buffer="./dataset/temptest/words_ids.csv",encoding="utf-8")
print(df_words_ids.head(5))
id2words=pd.Series(data=df_words_ids["words"].values,index=df_words_ids["id"].values)
print(id2words[2])

#读取words和ids的dataframe
    #df_words_ids=pd.read_csv(filepath_or_buffer="./dataset/"+name+"/words_ids.csv",encoding="utf-8")
    #读取tags和ids的dataframe
    #df_tags_ids=pd.read_csv(filepath_or_buffer="./dataset/"+name+"/tags_ids.csv",encoding="utf-8")
    #装换为words2id, id2words, tags2id, id2tags
    #df_data=pd.DataFrame(data={})
    #words2id=pd.Series(data=df_words_ids["id"].values,index=df_words_ids["words"].values)
    #id2words=pd.Series(data=df_words_ids["words"].values,index=df_words_ids["id"].values)
#print(df_word_ids["2"])