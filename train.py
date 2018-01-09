import pandas as pd
import numpy as np
import tensorflow as tf
import temp_file

# 读数据
df_train_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_train.pkl")
df_validation_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_validation.pkl")
df_test_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_test.pkl")
X_train_pw = np.asarray(list(df_train_pw['X'].values),dtype=np.int32)
y_train_pw = np.asarray(list(df_train_pw['y'].values),dtype=np.int32)
print("shapa of X_train_pw:", X_train_pw.shape)
print("type of X_train_pw:", X_train_pw.dtype)
X_validation_pw = np.asarray(list(df_validation_pw['X'].values),dtype=np.int32)
y_validation_pw = np.asarray(list(df_validation_pw['y'].values),dtype=np.int32)
X_test_pw = np.asarray(list(df_test_pw['X'].values))
y_test_pw = np.asarray(list(df_test_pw['y'].values))

df_train_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_train.pkl")
df_validation_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_validation.pkl")
df_test_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_test.pkl")
X_train_pph = np.asarray(list(df_train_pph['X'].values),dtype=np.int32)
print("shapa of X_train_pph:",X_train_pph.shape)
print("type of X_train_pph:",X_train_pph.dtype)
y_train_pph = np.asarray(list(df_train_pph['y'].values),dtype=np.int32)
X_validation_pph = np.asarray(list(df_validation_pph['X'].values),dtype=np.int32)
y_validation_pph = np.asarray(list(df_validation_pph['y'].values),dtype=np.int32)
X_test_pph = np.asarray(list(df_test_pph['X'].values))
y_test_pph = np.asarray(list(df_test_pph['y'].values))

df_train_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_train.pkl")
df_validation_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_validation.pkl")
df_test_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_test.pkl")
X_train_iph = np.asarray(list(df_train_iph['X'].values),dtype=np.int32)
y_train_iph = np.asarray(list(df_train_iph['y'].values),dtype=np.int32)
print("shapa of X_train_iph:", X_train_iph.shape)
print("type of X_train_iph:", X_train_iph.dtype)
X_validation_iph = np.asarray(list(df_validation_iph['X'].values),dtype=np.int32)
y_validation_iph = np.asarray(list(df_validation_iph['y'].values),dtype=np.int32)
X_test_iph = np.asarray(list(df_test_iph['X'].values))
y_test_iph = np.asarray(list(df_test_iph['y'].values))

X_train=[X_train_pw,X_train_pph,X_train_iph]
y_train=[y_train_pw,y_train_pph,y_train_iph]
X_validation=[X_validation_pw,X_validation_pph,X_validation_iph]
y_validation=[y_validation_pw,y_validation_pph,y_validation_iph]

model = temp_file.Seq2Seq()
model.fit(X_train, y_train, X_validation, y_validation, "test", False)

