import numpy as np
import os

# train && test
if __name__ == "__main__":
    # 读数据
    df_train_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_train.pkl")
    df_validation_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_validation.pkl")
    X_train_pw = np.asarray(list(df_train_pw['X'].values))
    y_train_pw = np.asarray(list(df_train_pw['y'].values))
    X_validation_pw = np.asarray(list(df_validation_pw['X'].values))
    y_validation_pw = np.asarray(list(df_validation_pw['y'].values))

    df_train_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_train.pkl")
    df_validation_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_validation.pkl")
    X_train_pph = np.asarray(list(df_train_pph['X'].values))
    y_train_pph = np.asarray(list(df_train_pph['y'].values))
    X_validation_pph = np.asarray(list(df_validation_pph['X'].values))
    y_validation_pph = np.asarray(list(df_validation_pph['y'].values))

    df_train_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_train.pkl")
    df_validation_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_validation.pkl")
    X_train_iph = np.asarray(list(df_train_iph['X'].values))
    y_train_iph = np.asarray(list(df_train_iph['y'].values))
    X_validation_iph = np.asarray(list(df_validation_iph['X'].values))
    y_validation_iph = np.asarray(list(df_validation_iph['y'].values))

    X_train = [X_train_pw, X_train_pph, X_train_iph]
    y_train = [y_train_pw, y_train_pph, y_train_iph]
    X_validation = [X_validation_pw, X_validation_pph, X_validation_iph]
    y_validation = [y_validation_pw, y_validation_pph, y_validation_iph]

    print("X_train_pw:\n", X_train_pw);
    print(X_train_pw.shape)
    print("X_train_pph:\n", X_train_pph);
    print(X_train_pph.shape)
    print("X_train_iph:\n", X_train_iph);
    print(X_train_iph.shape)

    model = Attension_Alignment_Seq2Seq()
    model.fit(X_train, y_train, X_validation, y_validation, "test", False)

'''

file_train=open(file="./data/corpus/xiekun.out.train.txt",encoding="utf-8")
doc_train=""
lines_train=file_train.readlines()
for line_train in lines_train:
    doc_train+=line_train
print(len(lines_train))

file_test=open(file="./data/corpus/xiekun.out.test.txt",encoding="utf-8")
doc_test=""
lines_test=file_test.readlines()
for line_test in lines_test:
    doc_test+=line_test
print(len(lines_test))


doc=doc_train+doc_test
f=open(file="./data/corpus/prosody.txt",mode="w",encoding="utf-8")
f.write(doc)
f.close()
'''



