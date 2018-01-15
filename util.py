import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import parameter

#compute accuracy,precison,recall and f1
def eval(y_true,y_pred):
    #accuracy
    accuracy=accuracy_score(y_true=y_true,y_pred=y_pred)

    #class 1
    binarized_y_true_1=binarize(sequence=y_true,positive_value=1)
    binarized_y_pred_1=binarize(sequence=y_pred,positive_value=1)
    recall_1=recall_score(y_true=binarized_y_true_1,y_pred=binarized_y_pred_1)
    precision_1=precision_score(y_true=binarized_y_true_1,y_pred=binarized_y_pred_1)
    f_1=f1_score(y_true=binarized_y_true_1,y_pred=binarized_y_pred_1)

    # class 2
    binarized_y_true_2 = binarize(sequence=y_true, positive_value=2)
    binarized_y_pred_2 = binarize(sequence=y_pred, positive_value=2)
    recall_2 = recall_score(y_true=binarized_y_true_2, y_pred=binarized_y_pred_2)
    precision_2 = precision_score(y_true=binarized_y_true_2, y_pred=binarized_y_pred_2)
    f_2 = f1_score(y_true=binarized_y_true_2, y_pred=binarized_y_pred_2)

    return accuracy,f_1,f_2

#以positive_value为正类别,来二值化一个sequence.计算metrics用到
def binarize(sequence,positive_value):
    #deep copy
    temp_sequence=sequence.copy()
    temp_sequence[temp_sequence!=positive_value]=0
    temp_sequence[temp_sequence==positive_value]=1
    return temp_sequence


#recover to original result
def recover(X,preds_pw,preds_pph,preds_iph,filename):
    #get complex "#" index
    length=preds_pw.shape[0]
    complex=np.array([preds_iph,preds_pph,preds_pw])
    arg = np.argmax(complex, axis=0)
    #print("arg:\n", arg)
    for i in range(length):
        if arg[i] == 0:
            if complex[0, i] == 2:
                arg[i] = 6
            else:
                arg[i] = 0
        if arg[i] == 1:
            if complex[1, i] == 2:
                arg[i] = 4
            else:
                arg[i] = 0
        if arg[i] == 2:
            if complex[2, i] == 2:
                arg[i] = 2
            else:
                arg[i] = 0
    arg = (arg / 2).astype(dtype=np.int32)
    #shape of arg:[test_size,max_sentence_size]
    arg=np.reshape(arg,newshape=(-1,parameter.MAX_SENTENCE_SIZE))
    #print("arg.shape",arg.shape)
    #print("arg:\n", arg)
    #get id2words
    df_words_ids = pd.read_csv(filepath_or_buffer="./dataset/temptest/words_ids.csv", encoding="utf-8")
    #print(df_words_ids.head(5))
    id2words = pd.Series(data=df_words_ids["words"].values, index=df_words_ids["id"].values)
    #print(id2words[2])
    doc=""
    for i in range(X.shape[0]):
        sentence=""
        for j in range(X.shape[1]):
            if(X[i][j])==0:
                break;
            else:
                sentence+=id2words[X[i][j]]
                if(arg[i][j]!=0):
                    sentence+=("#"+str(arg[i][j]))
        sentence+="\n"
        doc+=sentence
    f=open(filename,mode="w",encoding="utf-8")
    f.write(doc)
    f.close()

if __name__ =="__main__":
    #测试

    a=np.array([1,2,3,4,0,5,6,7,1,1,2,1,0])
    print(a)
    result=binarize(sequence=a,positive_value=1)
    print(result)
    print(a)





