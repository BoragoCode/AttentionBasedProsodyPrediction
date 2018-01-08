'''
    model that without attension
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import time
import os
import parameter

class Seq2Seq():
    def __init__(self):
        # basic environment
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        #basic parameters
        self.learning_rate = parameter.LEARNING_RATE
        self.max_epoch = parameter.MAX_EPOCH
        self.embedding_size = parameter.EMBEDDING_SIZE
        self.class_num = parameter.CLASS_NUM
        self.hidden_units_num = parameter.HIDDEN_UNITS_NUM
        self.hidden_units_num2=parameter.HIDDEN_UNITS_NUM2
        self.layer_num = parameter.LAYER_NUM
        self.max_sentence_size=parameter.MAX_SENTENCE_SIZE
        self.vocab_size=parameter.VOCAB_SIZE
        self.batch_size=parameter.BATCH_SIZE

    # display accuracy and loss information
    def showInfo(self, print_log=False, loss=None, accuracy=None, train_losses=None, train_accus=None):
        if print_log:
            print("----training loss  : ", loss)
            print("----train accuracy : ", accuracy)
            print()
        else:
            print("----average training loss       : ", sum(train_losses) / len(train_losses))
            print("----average training accuracy   : ", sum(train_accus) / len(train_accus))
            print("----average validation loss     : ", loss)
            print("----average validation accuracy : ", accuracy)

    #encoder,传入是前向和后向的cell,还有inputs
    #输出是
    def encoder(self,cell_forward,cell_backward,inputs):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_forward,
            cell_bw=cell_backward,
            inputs=inputs,
            dtype=tf.float32
        )
        print("shape of states:",states)
        outputs_forward = outputs[0]   # shape of h is [batch_size, max_time, cell_fw.output_size]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]
        #shape of h is [batch_size, max_time, cell_fw.output_size*2]
        encoder_outputs = tf.concat(values=[outputs_forward, outputs_backward], axis=2)

        states_forward=states[0]       # .c:[batch_size,cell_fw.output_size]   .h:[batch_size,cell_fw.output_size]
        states_backward=states[1]
        #print(type(states_forward))
        #shape of encoder_states_concat[2,batch_size,cell_fw.output_size*2]
        #encoder_states_concat = tf.concat([states_forward, states_backward], axis=2)
        #print(encoder_states_concat)
        #encoder_states=(encoder_states_concat[0],encoder_states_concat[1])
        #encoder_states=tuple(encoder_states)
        #print(type(encoder_states))
        return encoder_outputs,states_forward

    def decoder(self,cell,initial_state,inputs):
        outputs,states=tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            initial_state=initial_state
        )
        #outputs      #[batch_size,time_steps,hidden_size*2]
        decoder_outputs=tf.reshape(tensor=outputs,shape=(-1,self.hidden_units_num))
        return decoder_outputs

        '''
        decoder=seq2seq.BasicDecoder(
            cell=cell,initial_state=initial_state
        )
        '''

    # forward process and training process
    def fit(self,X_train,y_train,X_validation,y_validation,name,print_log=True):
        #---------------------------------------forward computation--------------------------------------------#
        X_train_pw=X_train[0];  X_train_pph=X_train[1]; X_train_iph=X_train[2]
        y_train_pw = y_train[0];y_train_pph = y_train[1];y_train_iph = X_train[2]

        X_validation_pw = X_validation[0];X_validation_pph = X_validation[1];X_validation_iph = X_validation[2]
        y_validation_pw = y_validation[0];y_validation_pph = y_validation[1];y_validation_iph = y_validation[2]

        #---------------------------------------define graph---------------------------------------------#
        with self.graph.as_default():
            # data place holder
            self.X_p_pw = tf.placeholder(
                    dtype=tf.int32,
                    shape=(None, self.max_sentence_size),
                    name="input_placeholder"
            )
            self.y_p_pw = tf.placeholder(
                    dtype=tf.int32,
                    shape=(None,self.max_sentence_size),
                    name="label_placeholder"
            )

            self.X_p_pph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="input_placeholder"
            )
            self.y_p_pph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="label_placeholder"
            )

            self.X_p_iph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="input_placeholder"
            )
            self.y_p_iph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="label_placeholder"
            )

            #n
            self.class1_p = tf.placeholder(
                dtype=tf.int32,
                shape=(None,),
                name="class_1"
            )
            #b
            self.class2_p = tf.placeholder(
                dtype=tf.int32,
                shape=(None,),
                name="class_2"
            )

            # embeddings
            embeddings = tf.Variable(
                initial_value=tf.zeros(shape=(self.vocab_size, self.embedding_size), dtype=tf.float32),
                name="embeddings"
            )

            # ---------------------------------/hierarchy:PPH/-------------------------------------------#
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pph = tf.nn.embedding_lookup(params=embeddings, ids=self.X_p_pph, name="embeded_input_pph")
            #this input use reuslts of pre stpes
            #shape of inputs[batch_size,max_time_stpes,embeddings_dims+class_num]
            #inputs_pph=tf.concat(values=[inputs_pph,pred_normal_one_hot_pw],axis=2,name="inputs_pph")
            print("shape of input_pph:",inputs_pph.shape)

            # encoder
            # forward part
            en_lstm_forward1_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            # backward part
            en_lstm_backward1_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            # decoder cells
            de_lstm_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)

            # encode
            #encoder_outputs_pph, encoder_states_pph = self.encoder(
            #    cell_forward=en_lstm_forward1_pph,
            #    cell_backward=en_lstm_backward1_pph,
            #    inputs=inputs_pph
            #)
            # shape of h is [batch*time_steps,hidden_units]
            #h_pph = self.decoder(cell=de_lstm_pph, initial_state=encoder_states_pph, inputs=encoder_outputs_pph)

            en_outputs_pph, en_states_pph = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=en_lstm_forward1_pph,
                cell_bw=en_lstm_backward1_pph,
                inputs=inputs_pph,
                dtype=tf.float32
            )

            # print("shape of states:", states_pph)
            outputs_forward_pph = en_outputs_pph[0]  # shape of h is [batch_size, max_time, cell_fw.output_size]
            outputs_backward_pph = en_outputs_pph[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]
            # shape of h is [batch_size, max_time, cell_fw.output_size*2]
            encoder_outputs_pph = tf.concat(values=[outputs_forward_pph, outputs_backward_pph], axis=2)
            states_forward_pph = en_states_pph[0]

            de_outputs_pph, de_states_pph = tf.nn.dynamic_rnn(
                cell=de_lstm_pph,
                inputs=encoder_outputs_pph,
                initial_state=states_forward_pph
            )

            # outputs      #[batch_size,time_steps,hidden_size*2]
            h_pph = tf.reshape(tensor=de_outputs_pph, shape=(-1, self.hidden_units_num))
            # return decoder_outputs

            # fully connect layer
            w_pph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num2, self.class_num)),
                name="weights_pph"
            )
            b_pph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_pph"
            )
            logits_pph = tf.matmul(h_pph, w_pph) + b_pph  # shape of logits:[batch_size*max_time, 5]

            # prediction
            # shape of pred[batch_size*max_time, 1]
            pred_pph = tf.cast(tf.argmax(logits_pph, 1), tf.int32, name="pred_pph")

            # pred in an normal way,shape is [batch_size, max_time,1]
            pred_normal_pph = tf.reshape(
                tensor=pred_pph,
                shape=(-1, self.max_sentence_size),
                name="pred_normal"
            )

            # one-hot the pred_normal:[batch_size, max_time,class_num]
            pred_normal_one_hot_pph = tf.one_hot(
                indices=pred_normal_pph,
                depth=self.class_num
            )

            # correct_prediction
            correct_prediction_pph = tf.equal(pred_pph, tf.reshape(self.y_p_pph, [-1]))
            # accracy
            self.accuracy_pph = tf.reduce_mean(
                input_tensor=tf.cast(x=correct_prediction_pph, dtype=tf.float32),
                name="accuracy_pph"
            )
            # class #1
            # class1=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=1)
            basic_class_1_pph = tf.cast(tf.equal(self.class1_p, tf.reshape(self.y_p_pph, [-1])), dtype=tf.int32)
            pred_class_1_pph = tf.cast(tf.equal(self.class1_p, tf.reshape(pred_pph, [-1])), dtype=tf.int32)
            correct_class_1_pph = tf.bitwise.bitwise_and(basic_class_1_pph, pred_class_1_pph)  # #1 prediction

            self.accuracy_class_1_pph = tf.divide(
                x=tf.reduce_sum(correct_class_1_pph),
                y=tf.reduce_sum(basic_class_1_pph),
                name="accuracy_class_1_pph"
            )

            # class #2
            # class2=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=2)
            basic_class_2_pph = tf.cast(tf.equal(self.class2_p, tf.reshape(self.y_p_pph, [-1])), dtype=tf.int32)
            pred_class_2_pph = tf.cast(tf.equal(self.class2_p, tf.reshape(pred_pph, [-1])), dtype=tf.int32)
            correct_class_2_pph = tf.bitwise.bitwise_and(basic_class_2_pph, pred_class_2_pph)  # #2 prediction
            self.accuracy_class_2_pph = tf.divide(
                x=tf.reduce_sum(correct_class_2_pph),
                y=tf.reduce_sum(basic_class_2_pph),
                name="accuracy_class_2_pph"
            )

            # loss
            self.loss_pph = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.reshape(self.y_p_pph, shape=[-1]),
                logits=logits_pph
            )
            # -------------------------------------------------------------------------------------------#

            '''
            # ---------------------------------/hierarchy:IPH/-------------------------------------------#
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_iph = tf.nn.embedding_lookup(params=embeddings, ids=self.X_p_iph, name="embeded_input_iph")
            # this input use reuslts of pre stpes
            # shape of inputs[batch_size,max_time_stpes,embeddings_dims+class_num]
            inputs_iph = tf.concat(values=[inputs_iph, pred_normal_one_hot_pph], axis=2, name="inputs_iph")

            # encoder
            # forward part
            en_lstm_forward1_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            # backward part
            en_lstm_backward1_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            # decoder cells
            de_lstm_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)

            # encode
            encoder_outputs_iph, encoder_states_iph = self.encoder(
                cell_forward=en_lstm_forward1_iph,
                cell_backward=en_lstm_backward1_iph,
                inputs=inputs_iph
            )
            # shape of h is [batch*time_steps,hidden_units]
            h_iph = self.decoder(cell=de_lstm_iph, initial_state=encoder_states_iph, inputs=encoder_outputs_iph)

            # fully connect layer
            w_iph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num2, self.class_num)),
                name="weights_iph"
            )
            b_iph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_iph"
            )
            logits_iph = tf.matmul(h_iph, w_iph) + b_iph  # shape of logits:[batch_size*max_time, 5]

            # prediction
            # shape of pred[batch_size*max_time, 1]
            pred_iph = tf.cast(tf.argmax(logits_iph, 1), tf.int32, name="pred_iph")

            # pred in an normal way,shape is [batch_size, max_time,1]
            pred_normal_iph = tf.reshape(
                tensor=pred_iph,
                shape=(-1, self.max_sentence_size),
                name="pred_normal"
            )

            # one-hot the pred_normal:[batch_size, max_time,class_num]
            pred_normal_one_hot_iph = tf.one_hot(
                indices=pred_normal_iph,
                depth=self.class_num
            )

            # correct_prediction
            correct_prediction_iph = tf.equal(pred_iph, tf.reshape(self.y_p_iph, [-1]))
            # accracy
            self.accuracy_iph = tf.reduce_mean(
                input_tensor=tf.cast(x=correct_prediction_iph, dtype=tf.float32),
                name="accuracy_iph"
            )
            # class #1
            # class1=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=1)
            basic_class_1_iph = tf.cast(tf.equal(self.class1_p, tf.reshape(self.y_p_iph, [-1])), dtype=tf.int32)
            pred_class_1_iph = tf.cast(tf.equal(self.class1_p, tf.reshape(pred_iph, [-1])), dtype=tf.int32)
            correct_class_1_iph = tf.bitwise.bitwise_and(basic_class_1_iph, pred_class_1_iph)  # #1 prediction

            self.accuracy_class_1_iph = tf.divide(
                x=tf.reduce_sum(correct_class_1_iph),
                y=tf.reduce_sum(basic_class_1_iph),
                name="accuracy_class_1_iph"
            )

            # class #2
            # class2=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=2)
            basic_class_2_iph = tf.cast(tf.equal(self.class2_p, tf.reshape(self.y_p_iph, [-1])), dtype=tf.int32)
            pred_class_2_iph = tf.cast(tf.equal(self.class2_p, tf.reshape(pred_iph, [-1])), dtype=tf.int32)
            correct_class_2_iph = tf.bitwise.bitwise_and(basic_class_2_iph, pred_class_2_iph)  # #2 prediction
            self.accuracy_class_2_iph = tf.divide(
                x=tf.reduce_sum(correct_class_2_iph),
                y=tf.reduce_sum(basic_class_2_iph),
                name="accuracy_class_2_iph"
            )

            # loss
            self.loss_iph = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.reshape(self.y_p_iph, shape=[-1]),
                logits=logits_iph
            )
            # --------------------------------------------------------------------------------------------#
            '''

            #loss
            self.loss=self.loss_pph            #+self.loss_iph
            #optimizer
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.init_op=tf.global_variables_initializer()

        #------------------------------------/*Session*/-----------------------------------------
        with self.session as sess:
            print("Training Start")
            sess.run(self.init_op)          #initialize all variables

            train_Size = X_train_pw.shape[0];
            validation_Size = X_validation_pw.shape[0]
            best_validation_accuracy = 0        #best validation accuracy in training process

            for epoch in range(1,self.max_epoch+1):
                print("Epoch:", epoch)
                start_time=time.time()      #time evaluation
                train_losses_pw = [];  train_accus_pw = []       # training loss/accuracy in every mini-batch
                #c1_accus = [];  c2_accus = [];              # each class's accuracy
                # mini batch
                for i in range(0, (train_Size // self.batch_size)):
                    _, train_loss, train_accuracy_pw = sess.run(
                        fetches=[self.optimizer, self.loss, self.accuracy_pph],
                        feed_dict={
                            self.X_p_pw:X_train_pw[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_pw:y_train_pw[i * self.batch_size:(i + 1) * self.batch_size],
                            self.X_p_pph: X_train_pph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_pph: y_train_pph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.X_p_iph: X_train_iph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_iph: y_train_iph[i * self.batch_size:(i + 1) * self.batch_size],
                        }
                    )
                    print("train_loss:",train_loss)
                    print("train_accuracy_pw:",train_accuracy_pw)
                    '''
                    c1, c2= sess.run(
                        fetches=[self.accuracy_class_1,
                                 self.accuracy_class_2],
                        feed_dict={
                            self.X_p: X_train[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p: y_train[i * self.batch_size:(i + 1) * self.batch_size],
                            self.class1_p: np.full(shape=(self.batch_size * self.max_sentence_size,), fill_value=1),
                            self.class2_p: np.full(shape=(self.batch_size * self.max_sentence_size,), fill_value=2)
                        }
                    )
                    # print training infomation
                    if (print_log):
                        self.showInfo(print_log,train_loss,train_accuracy)
                    # add to list
                    train_losses.append(train_loss); train_accus.append(train_accuracy)
                    c1_accus.append(c1);    c2_accus.append(c2)

                validation_loss, validation_accuracy, c1_va, c2_va = sess.run(
                    fetches=[self.loss,
                             self.accuracy,
                             self.accuracy_class_1,
                             self.accuracy_class_2],
                    feed_dict={
                        self.X_p: X_validation,
                        self.y_p: y_validation,
                        self.class1_p: np.full(shape=(X_validation.shape[0] * self.max_sentence_size,), fill_value=1),
                        self.class2_p: np.full(shape=(X_validation.shape[0] * self.max_sentence_size,), fill_value=2)
                    }
                )
                print("Epoch ",epoch," finished.","spend ",round((time.time()-start_time)/60,2)," mins")
                self.showInfo(False, validation_loss, validation_accuracy, train_losses, train_accus)
                print("----average training accuracy of class 1:", sum(c1_accus) / len(c1_accus))
                print("----average training accuracy of class 2:", sum(c2_accus) / len(c2_accus))
                print("----average validation accuracy of class 1:", c1_va)
                print("----average validation accuracy of class 2:", c2_va)

                # when we get a new best validation accuracy,we store the model
                if best_validation_accuracy < validation_accuracy:
                    print("New Best Accuracy ",validation_accuracy," On Validation set! ")
                    print("Saving Models......")
                    #exist ./models folder?
                    if not os.path.exists("./models/"):
                        os.mkdir(path="./models/")
                    if not os.path.exists("./models/"+name):
                        os.mkdir(path="./models/"+name)
                    if not os.path.exists("./models/"+name+"/bilstm"):
                        os.mkdir(path="./models/"+name+"/bilstm")
                    #create saver
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/"+name+"/bilstm/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/"+name+"/bilstm/my-model-10000.meta")
                '''

    #返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
    def pred(self,name,X,y=None,):
        start_time = time.time()    #compute time
        if y is None:
            with self.session as sess:
                # restore model
                new_saver=tf.train.import_meta_graph(
                    meta_graph_or_file="./models/"+name+"/bilstm/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/"+name+"/bilstm/my-model-10000")
                #get default graph
                graph = tf.get_default_graph()
                # get opration from the graph
                pred_normal = graph.get_operation_by_name("pred_normal").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                pred = sess.run(fetches=pred_normal, feed_dict={X_p: X})
                print("this operation spends ",round((time.time()-start_time)/60,2)," mins")
                return pred
        else:
            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
                graph = tf.get_default_graph()
                # get opration from the graph
                accuracy=graph.get_operation_by_name("accuracy").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                y_p=graph.get_operation_by_name("label_placeholder").outputs[0]
                #forward and get the results
                accu = sess.run(fetches=accuracy,feed_dict={X_p: X,y_p: y})
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
                return accu

    #把一个句子转成一个分词后的结构
    def infer(self,sentence,name):
        pass

#train && test
if __name__=="__main__":
    # 读数据
    df_train_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_train.pkl")
    df_validation_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_validation.pkl")
    df_test_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_test.pkl")
    X_train_pw = np.asarray(list(df_train_pw['X'].values))
    y_train_pw = np.asarray(list(df_train_pw['y'].values))
    X_validation_pw = np.asarray(list(df_validation_pw['X'].values))
    y_validation_pw = np.asarray(list(df_validation_pw['y'].values))
    X_test_pw = np.asarray(list(df_test_pw['X'].values))
    y_test_pw = np.asarray(list(df_test_pw['y'].values))

    df_train_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_train.pkl")
    df_validation_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_validation.pkl")
    df_test_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_test.pkl")
    X_train_pph = np.asarray(list(df_train_pph['X'].values))
    y_train_pph = np.asarray(list(df_train_pph['y'].values))
    X_validation_pph = np.asarray(list(df_validation_pph['X'].values))
    y_validation_pph = np.asarray(list(df_validation_pph['y'].values))
    X_test_pph = np.asarray(list(df_test_pph['X'].values))
    y_test_pph = np.asarray(list(df_test_pph['y'].values))

    df_train_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_train.pkl")
    df_validation_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_validation.pkl")
    df_test_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_test.pkl")
    X_train_iph = np.asarray(list(df_train_iph['X'].values))
    y_train_iph = np.asarray(list(df_train_iph['y'].values))
    X_validation_iph = np.asarray(list(df_validation_iph['X'].values))
    y_validation_iph = np.asarray(list(df_validation_iph['y'].values))
    X_test_iph = np.asarray(list(df_test_iph['X'].values))
    y_test_iph = np.asarray(list(df_test_iph['y'].values))

    X_train=[X_train_pw,X_test_pph,X_train_iph]
    y_train=[y_train_pw,y_train_pph,y_train_iph]
    X_validation=[X_validation_pw,X_validation_pph,X_validation_iph]
    y_validation=[y_validation_pw,y_validation_pph,y_validation_iph]

    # train model
    model = Seq2Seq()
    model.fit(X_train, y_train, X_validation, y_validation, "test", False)

    # testing model
    # accuracy = model.pred(name="test", X=X_test, y=y_test)
    # print(accuracy)