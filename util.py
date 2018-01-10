import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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

#以positive_value为正类别,来二值化一个sequence.
def binarize(sequence,positive_value):
    #deep copy
    temp_sequence=sequence.copy()
    temp_sequence[temp_sequence!=positive_value]=0
    temp_sequence[temp_sequence==positive_value]=1
    return temp_sequence

if __name__ =="__main__":
    a=np.array([1,2,3,4,0,5,6,7,1,1,2,1,0])
    print(a)
    result=binarize(sequence=a,positive_value=1)
    print(result)
    print(a)





