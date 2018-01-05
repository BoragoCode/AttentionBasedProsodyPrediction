'''
    ####file that contains LSTM parameters of this model
    ####modify this file to change LSTM parameters
'''

#can modify
LEARNING_RATE=0.001                 #学习率
MAX_SENTENCE_SIZE=24                #固定句子长度为24
TIMESTEP_SIZE=MAX_SENTENCE_SIZE     #LSTM的time_step应该和句子长度一致
INPUT_SIZE=EMBEDDING_SIZE=1001      #嵌入向量维度,和输入大小应当一样
DECAY=0.85
MAX_EPOCH=30                        #最大迭代次数
LAYER_NUM=2                         #lstm层数2
HIDDEN_UNITS_NUM=128                #隐藏层结点数量
HIDDEN_UNITS_NUM2=128               #隐藏层2结点数量
BATCH_SIZE=1024                     #batch大小
DROPOUT_RATE=0.5                    #dropout 比率


#can't modify
CLASS_NUM=3                         #类别数量
VOCAB_SIZE=4711                     # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到

