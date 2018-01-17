'''
    model with attention
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import time
import os
import parameter
import util

class Alignment_Seq2Seq():
    def __init__(self):
        # basic environment
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        # basic parameters
        self.learning_rate = parameter.LEARNING_RATE
        self.max_epoch = parameter.MAX_EPOCH
        self.embedding_size = parameter.EMBEDDING_SIZE
        self.class_num = parameter.CLASS_NUM
        self.hidden_units_num = parameter.HIDDEN_UNITS_NUM
        self.hidden_units_num2 = parameter.HIDDEN_UNITS_NUM2
        self.layer_num = parameter.LAYER_NUM
        self.max_sentence_size = parameter.MAX_SENTENCE_SIZE
        self.vocab_size = parameter.VOCAB_SIZE
        self.batch_size = parameter.BATCH_SIZE
        self.lambda_pw=parameter.LAMBDA_PW
        self.lambda_pph=parameter.LAMBDA_PPH
        self.lambda_iph=parameter.LAMBDA_IPH

    # encoder,传入是前向和后向的cell,还有inputs
    # 输出是
    def encoder(self, cell_forward, cell_backward, inputs, seq_length, scope_name):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_forward,
            cell_bw=cell_backward,
            inputs=inputs,
            sequence_length=seq_length,
            dtype=tf.float32,
            scope=scope_name
        )

        outputs_forward = outputs[0]  # shape of h is [batch_size, max_time, cell_fw.output_size]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]
        states_forward = states[0]  # .c:[batch_size,num_units]   .h:[batch_size,num_units]
        states_backward = states[1]
        #concat final outputs [batch_size, max_time, cell_fw.output_size*2]
        encoder_outputs = tf.concat(values=[outputs_forward, outputs_backward], axis=2)
        #concat final states
        state_h_concat=tf.concat(values=[states_forward.h,states_backward.h],axis=1,name="state_h_concat")
        #print("state_h_concat:",state_h_concat)
        state_c_concat=tf.concat(values=[states_forward.c,states_backward.c],axis=1,name="state_c_concat")
        #print("state_c_concat:",state_c_concat)
        encoder_states=rnn.LSTMStateTuple(c=state_c_concat,h=state_h_concat)

        return encoder_outputs, encoder_states

    def decoder(self, cell, initial_state, inputs, scope_name):
        # outputs:[batch_size,time_steps,hidden_size*2]
        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            initial_state=initial_state,
            scope=scope_name
        )
        #[batch_size*time_steps,hidden_size*2]
        decoder_outputs = tf.reshape(tensor=outputs, shape=(-1, self.hidden_units_num*2))
        return decoder_outputs

    # forward process and training process
    def fit(self, X_train, y_train, len_train, X_validation, y_validation, len_validation, name, print_log=True):
        # ---------------------------------------forward computation--------------------------------------------#
        y_train_pw = y_train[0]
        y_train_pph = y_train[1]
        y_train_iph = y_train[2]

        y_validation_pw = y_validation[0]
        y_validation_pph = y_validation[1]
        y_validation_iph = y_validation[2]
        # ---------------------------------------define graph---------------------------------------------#
        with self.graph.as_default():
            # data place holder
            self.X_p = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="input_placeholder"
            )

            self.y_p_pw = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="label_placeholder_pw"
            )
            self.y_p_pph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="label_placeholder_pph"
            )
            self.y_p_iph = tf.placeholder(
                dtype=tf.int32,
                shape=(None, self.max_sentence_size),
                name="label_placeholder_iph"
            )

            # 相应序列的长度占位
            self.seq_len_p = tf.placeholder(
                dtype=tf.int32,
                shape=(None,),
                name="seq_len"
            )

            #用来去掉padding的mask
            self.mask = tf.sequence_mask(
                lengths=self.seq_len_p,
                maxlen=self.max_sentence_size,
                name="mask"
            )

            #去掉padding之后的labels
            y_p_pw_masked = tf.boolean_mask(                #shape[seq_len1+seq_len2+....+,]
                tensor=self.y_p_pw,
                mask=self.mask,
                name="y_p_pw_masked"
            )
            y_p_pph_masked = tf.boolean_mask(               # shape[seq_len1+seq_len2+....+,]
                tensor=self.y_p_pph,
                mask=self.mask,
                name="y_p_pph_masked"
            )
            y_p_iph_masked = tf.boolean_mask(               # shape[seq_len1+seq_len2+....+,]
                tensor=self.y_p_iph,
                mask=self.mask,
                name="y_p_iph_masked"
            )

            # embeddings
            self.embeddings = tf.Variable(
                initial_value=tf.zeros(shape=(self.vocab_size, self.embedding_size), dtype=tf.float32),
                name="embeddings"
            )

            # -------------------------------------PW-----------------------------------------------------
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pw = tf.nn.embedding_lookup(params=self.embeddings, ids=self.X_p, name="embeded_input_pw")

            # encoder cells
            # forward part
            en_lstm_forward1_pw = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            # backward part
            en_lstm_backward1_pw = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            # decoder cells
            de_lstm_pw = rnn.BasicLSTMCell(num_units=self.hidden_units_num*2)

            # encode
            encoder_outputs_pw, encoder_states_pw = self.encoder(
                cell_forward=en_lstm_forward1_pw,
                cell_backward=en_lstm_backward1_pw,
                inputs=inputs_pw,
                seq_length=self.seq_len_p,
                scope_name="en_lstm_pw"
            )
            # decode
            h_pw = self.decoder(                    # shape of h is [batch*time_steps,hidden_units*2]
                cell=de_lstm_pw,
                initial_state=encoder_states_pw,
                inputs=encoder_outputs_pw,
                scope_name="de_lstm_pw"
            )

            # fully connect layer(projection)
            w_pw = tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num*2, self.class_num)),
                name="weights_pw"
            )
            b_pw = tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_pw"
            )
            #logits
            logits_pw = tf.matmul(h_pw, w_pw) + b_pw        #logits_pw:[batch_size*max_time, 3]
            logits_normal_pw=tf.reshape(                    #logits in an normal way:[batch_size,max_time_stpes,3]
                tensor=logits_pw,
                shape=(-1,self.max_sentence_size,3),
                name="logits_normal_pw"
            )
            logits_pw_masked = tf.boolean_mask(             # logits_pw_masked [seq_len1+seq_len2+....+,3]
                tensor=logits_normal_pw,
                mask=self.mask,
                name="logits_pw_masked"
            )

            # prediction
            pred_pw = tf.cast(tf.argmax(logits_pw, 1), tf.int32, name="pred_pw")   # pred_pw:[batch_size*max_time,]
            pred_normal_pw = tf.reshape(                    # pred in an normal way,[batch_size, max_time]
                tensor=pred_pw,
                shape=(-1, self.max_sentence_size),
                name="pred_normal_pw"
            )

            pred_pw_masked = tf.boolean_mask(  # logits_pw_masked [seq_len1+seq_len2+....+,]
                tensor=pred_normal_pw,
                mask=self.mask,
                name="pred_pw_masked"
            )

            pred_normal_one_hot_pw = tf.one_hot(            # one-hot the pred_normal:[batch_size, max_time,class_num]
                indices=pred_normal_pw,
                depth=self.class_num,
                name="pred_normal_one_hot_pw"
            )

            # loss
            self.loss_pw = tf.losses.sparse_softmax_cross_entropy(
                labels=y_p_pw_masked,
                logits=logits_pw_masked
            )+tf.contrib.layers.l2_regularizer(self.lambda_pw)(w_pw)
            # ---------------------------------------------------------------------------------------

            # ----------------------------------PPH--------------------------------------------------
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pph = tf.nn.embedding_lookup(params=self.embeddings, ids=self.X_p, name="embeded_input_pph")
            # shape of inputs[batch_size,max_time_stpes,embeddings_dims+class_num]
            inputs_pph = tf.concat(values=[inputs_pph, pred_normal_one_hot_pw], axis=2, name="inputs_pph")
            # print("shape of input_pph:", inputs_pph.shape)

            # encoder cells
            # forward part
            en_lstm_forward1_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            # backward part
            en_lstm_backward1_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            # decoder cells
            de_lstm_pph = rnn.BasicLSTMCell(num_units=self.hidden_units_num*2)

            # encode
            encoder_outputs_pph, encoder_states_pph = self.encoder(
                cell_forward=en_lstm_forward1_pph,
                cell_backward=en_lstm_backward1_pph,
                inputs=inputs_pph,
                seq_length=self.seq_len_p,
                scope_name="en_lstm_pph"
            )
            # shape of h is [batch*time_steps,hidden_units*2]
            h_pph = self.decoder(
                cell=de_lstm_pph,
                initial_state=encoder_states_pph,
                inputs=encoder_outputs_pph,
                scope_name="de_lstm_pph"
            )

            # fully connect layer(projection)
            w_pph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num*2, self.class_num)),
                name="weights_pph"
            )
            b_pph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_pph"
            )
            # logits
            logits_pph = tf.matmul(h_pph, w_pph) + b_pph  # shape of logits:[batch_size*max_time, 3]
            logits_normal_pph = tf.reshape(                 # logits in an normal way:[batch_size,max_time_stpes,3]
                tensor=logits_pph,
                shape=(-1, self.max_sentence_size, 3),
                name="logits_normal_pph"
            )
            logits_pph_masked = tf.boolean_mask(            # [seq_len1+seq_len2+....+,3]
                tensor=logits_normal_pph,
                mask=self.mask,
                name="logits_pph_masked"
            )

            # prediction
            pred_pph = tf.cast(tf.argmax(logits_pph, 1), tf.int32, name="pred_pph")  # pred_pph:[batch_size*max_time,]
            pred_normal_pph = tf.reshape(                       # pred in an normal way,[batch_size, max_time]
                tensor=pred_pph,
                shape=(-1, self.max_sentence_size),
                name="pred_normal_pph"
            )
            pred_pph_masked = tf.boolean_mask(                  # logits_pph_masked [seq_len1+seq_len2+....+,]
                tensor=pred_normal_pph,
                mask=self.mask,
                name="pred_pph_masked"
            )
            pred_normal_one_hot_pph = tf.one_hot(               # one-hot the pred_normal:[batch_size, max_time,class_num]
                indices=pred_normal_pph,
                depth=self.class_num,
                name="pred_normal_one_hot_pph"
            )

            # loss
            self.loss_pph = tf.losses.sparse_softmax_cross_entropy(
                labels=y_p_pph_masked,
                logits=logits_pph_masked
            )+tf.contrib.layers.l2_regularizer(self.lambda_pph)(w_pph)
            # ------------------------------------------------------------------------------------

            # ---------------------------------------IPH------------------------------------------
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_iph = tf.nn.embedding_lookup(params=self.embeddings, ids=self.X_p, name="embeded_input_iph")
            # shape of inputs[batch_size,max_time_stpes,embeddings_dims+class_num]
            inputs_iph = tf.concat(values=[inputs_iph, pred_normal_one_hot_pph], axis=2, name="inputs_pph")
            # print("shape of input_pph:", inputs_pph.shape)
            # encoder cells
            # forward part
            en_lstm_forward1_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            # backward part
            en_lstm_backward1_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            # en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            # en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            # decoder cells
            de_lstm_iph = rnn.BasicLSTMCell(num_units=self.hidden_units_num*2)

            # encode
            encoder_outputs_iph, encoder_states_iph = self.encoder(
                cell_forward=en_lstm_forward1_iph,
                cell_backward=en_lstm_backward1_iph,
                inputs=inputs_iph,
                seq_length=self.seq_len_p,
                scope_name="en_lstm_iph"
            )
            # shape of h is [batch*time_steps,hidden_units*2]
            h_iph = self.decoder(
                cell=de_lstm_iph,
                initial_state=encoder_states_iph,
                inputs=encoder_outputs_iph,
                scope_name="de_lstm_iph"
            )

            # fully connect layer(projection)
            w_iph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num*2, self.class_num)),
                name="weights_iph"
            )
            b_iph = tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_iph"
            )
            # logits
            logits_iph = tf.matmul(h_iph, w_iph) + b_iph  # shape of logits:[batch_size*max_time, 3]
            logits_normal_iph = tf.reshape(                # logits in an normal way:[batch_size,max_time_stpes,3]
                tensor=logits_iph,
                shape=(-1, self.max_sentence_size, 3),
                name="logits_normal_iph"
            )
            logits_iph_masked = tf.boolean_mask(  # [seq_len1+seq_len2+....+,3]
                tensor=logits_normal_iph,
                mask=self.mask,
                name="logits_iph_masked"
            )

            # prediction
            pred_iph = tf.cast(tf.argmax(logits_iph, 1), tf.int32, name="pred_iph")  # pred_iph:[batch_size*max_time,]
            pred_normal_iph = tf.reshape(  # pred in an normal way,[batch_size, max_time]
                tensor=pred_iph,
                shape=(-1, self.max_sentence_size),
                name="pred_normal_iph"
            )
            pred_iph_masked = tf.boolean_mask(  # logits_iph_masked [seq_len1+seq_len2+....+,]
                tensor=pred_normal_iph,
                mask=self.mask,
                name="pred_iph_masked"
            )
            pred_normal_one_hot_iph = tf.one_hot(  # one-hot the pred_normal:[batch_size, max_time,class_num]
                indices=pred_normal_iph,
                depth=self.class_num,
                name="pred_normal_one_hot_iph"
            )
            # loss
            self.loss_iph = tf.losses.sparse_softmax_cross_entropy(
                labels=y_p_iph_masked,
                logits=logits_iph_masked
            )+tf.contrib.layers.l2_regularizer(self.lambda_iph)(w_iph)

            # ---------------------------------------------------------------------------------------
            # loss
            self.loss = self.loss_pw + self.loss_pph + self.loss_iph
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()
            self.init_local_op = tf.local_variables_initializer()

        # ------------------------------------Session-----------------------------------------
        with self.session as sess:
            print("Training Start")
            sess.run(self.init_op)                  # initialize all variables
            sess.run(self.init_local_op)

            train_Size = X_train.shape[0];
            validation_Size = X_validation.shape[0]
            self.best_validation_loss = 1000                # best validation accuracy in training process

            # epoch
            for epoch in range(1, self.max_epoch + 1):
                print("Epoch:", epoch)
                start_time = time.time()  # time evaluation
                # training loss/accuracy in every mini-batch
                self.train_losses = []
                self.train_accus_pw = []
                self.train_accus_pph = []
                self.train_accus_iph = []

                self.c1_f_pw = [];
                self.c2_f_pw = []  # each class's f1 score
                self.c1_f_pph = [];
                self.c2_f_pph = []
                self.c1_f_iph = [];
                self.c2_f_iph = []

                # mini batch
                for i in range(0, (train_Size // self.batch_size)):
                    #注意:这里获取的都是mask之后的值
                    _, train_loss, y_train_pw_masked,y_train_pph_masked,y_train_iph_masked,\
                    train_pred_pw, train_pred_pph, train_pred_iph = sess.run(
                        fetches=[self.optimizer, self.loss,
                                 y_p_pw_masked,y_p_pph_masked,y_p_iph_masked,
                                 pred_pw_masked, pred_pph_masked, pred_iph_masked],
                        feed_dict={
                            self.X_p: X_train[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_pw: y_train_pw[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_pph: y_train_pph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p_iph: y_train_iph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.seq_len_p: len_train[i * self.batch_size:(i + 1) * self.batch_size]
                        }
                    )

                    # loss
                    self.train_losses.append(train_loss)
                    # metrics

                    accuracy_pw, f1_1_pw, f1_2_pw = util.eval(y_true=y_train_pw_masked,y_pred=train_pred_pw)    # pw
                    accuracy_pph, f1_1_pph, f1_2_pph = util.eval(y_true=y_train_pph_masked,y_pred=train_pred_pph)   # pph
                    accuracy_iph, f1_1_iph, f1_2_iph = util.eval(y_true=y_train_iph_masked,y_pred=train_pred_iph)   # iph

                    self.train_accus_pw.append(accuracy_pw)
                    self.train_accus_pph.append(accuracy_pph)
                    self.train_accus_iph.append(accuracy_iph)
                    # F1-score
                    self.c1_f_pw.append(f1_1_pw);
                    self.c2_f_pw.append(f1_2_pw)
                    self.c1_f_pph.append(f1_1_pph);
                    self.c2_f_pph.append(f1_2_pph)
                    self.c1_f_iph.append(f1_1_iph);
                    self.c2_f_iph.append(f1_2_iph)

                # validation in every epoch
                self.validation_loss, y_valid_pw_masked,y_valid_pph_masked,y_valid_iph_masked,\
                valid_pred_pw, valid_pred_pph, valid_pred_iph = sess.run(
                    fetches=[self.loss, y_p_pw_masked,y_p_pph_masked,y_p_iph_masked,
                             pred_pw_masked, pred_pph_masked, pred_iph_masked],
                    feed_dict={
                        self.X_p: X_validation,
                        self.y_p_pw: y_validation_pw,
                        self.y_p_pph: y_validation_pph,
                        self.y_p_iph: y_validation_iph,
                        self.seq_len_p: len_validation
                    }
                )
                # print("valid_pred_pw.shape:",valid_pred_pw.shape)
                # print("valid_pred_pph.shape:",valid_pred_pph.shape)
                # print("valid_pred_iph.shape:",valid_pred_iph.shape)

                # metrics
                self.valid_accuracy_pw, self.valid_f1_1_pw, self.valid_f1_2_pw = util.eval(y_true=y_valid_pw_masked,y_pred=valid_pred_pw)
                self.valid_accuracy_pph, self.valid_f1_1_pph, self.valid_f1_2_pph = util.eval(y_true=y_valid_pph_masked,y_pred=valid_pred_pph)
                self.valid_accuracy_iph, self.valid_f1_1_iph, self.valid_f1_2_iph = util.eval(y_true=y_valid_iph_masked,y_pred=valid_pred_iph)

                #print information
                print("Epoch ", epoch, " finished.", "spend ", round((time.time() - start_time) / 60, 2), " mins")
                self.showInfo(type="training")
                self.showInfo(type="validation")

                # when we get a new best validation accuracy,we store the model
                if self.best_validation_loss < self.validation_loss:
                    self.best_validation_loss = self.validation_loss
                    print("New Best loss ", self.best_validation_loss, " On Validation set! ")
                    print("Saving Models......\n\n")
                    # exist ./models folder?
                    if not os.path.exists("./models/"):
                        os.mkdir(path="./models/")
                    if not os.path.exists("./models/" + name):
                        os.mkdir(path="./models/" + name)
                    if not os.path.exists("./models/" + name + "/bilstm"):
                        os.mkdir(path="./models/" + name + "/bilstm")
                    # create saver
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/" + name + "/bilstm/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/" + name + "/bilstm/my-model-10000.meta")
                print("\n\n")
                # test:using X_validation_pw
                test_pred_pw, test_pred_pph, test_pred_iph = sess.run(
                    fetches=[pred_pw, pred_pph, pred_iph],
                    feed_dict={
                        self.X_p: X_validation,
                        self.seq_len_p: len_validation
                    }
                )
                # recover to original corpus txt
                # shape of valid_pred_pw,valid_pred_pw,valid_pred_pw:[corpus_size*time_stpes]
                util.recover(
                    X=X_validation,
                    preds_pw=test_pred_pw,
                    preds_pph=test_pred_pph,
                    preds_iph=test_pred_iph,
                    filename="recover_epoch_" + str(epoch) + ".txt"
                )

    # 返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
    def pred(self, name, X, y=None, ):
        start_time = time.time()  # compute time
        if y is None:
            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
                # get default graph
                graph = tf.get_default_graph()
                # get opration from the graph
                pred_normal = graph.get_operation_by_name("pred_normal").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                pred = sess.run(fetches=pred_normal, feed_dict={X_p: X})
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
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
                accuracy = graph.get_operation_by_name("accuracy").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                y_p = graph.get_operation_by_name("label_placeholder").outputs[0]
                # forward and get the results
                accu = sess.run(fetches=accuracy, feed_dict={X_p: X, y_p: y})
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
                return accu


    def showInfo(self, type):
        if type == "training":
            # training information
            print("                             /**Training info**/")
            print("----avarage training loss:", sum(self.train_losses) / len(self.train_losses))
            print("PW:")
            print("----avarage accuracy:", sum(self.train_accus_pw) / len(self.train_accus_pw))
            print("----avarage f1-Score of N:", sum(self.c1_f_pw) / len(self.c1_f_pw))
            print("----avarage f1-Score of B:", sum(self.c2_f_pw) / len(self.c2_f_pw))
            print("PPH:")
            print("----avarage accuracy :", sum(self.train_accus_pph) / len(self.train_accus_pph))
            print("----avarage f1-Score of N:", sum(self.c1_f_pph) / len(self.c1_f_pph))
            print("----avarage f1-Score of B:", sum(self.c2_f_pph) / len(self.c2_f_pph))
            print("IPH:")
            print("----avarage accuracy:", sum(self.train_accus_iph) / len(self.train_accus_iph))
            print("----avarage f1-Score of N:", sum(self.c1_f_iph) / len(self.c1_f_iph))
            print("----avarage f1-Score of B:", sum(self.c2_f_iph) / len(self.c2_f_iph))
        else:
            print("                             /**Validation info**/")
            print("----avarage validation loss:", self.validation_loss)
            print("PW:")
            print("----avarage accuracy:", self.valid_accuracy_pw)
            print("----avarage f1-Score of N:", self.valid_f1_1_pw)
            print("----avarage f1-Score of B:", self.valid_f1_2_pw)
            print("PPH:")
            print("----avarage accuracy :", self.valid_accuracy_pph)
            print("----avarage f1-Score of N:", self.valid_f1_1_pph)
            print("----avarage f1-Score of B:", self.valid_f1_2_pph)
            print("IPH:")
            print("----avarage accuracy:", self.valid_accuracy_iph)
            print("----avarage f1-Score of N:", self.valid_f1_1_iph)
            print("----avarage f1-Score of B:", self.valid_f1_2_iph)


# train && test
if __name__ == "__main__":
    # 读数据
    # pw
    df_train_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_train.pkl")
    df_validation_pw = pd.read_pickle(path="./dataset/temptest/pw_summary_validation.pkl")
    # pph
    df_train_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_train.pkl")
    df_validation_pph = pd.read_pickle(path="./dataset/temptest/pph_summary_validation.pkl")
    # iph
    df_train_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_train.pkl")
    df_validation_iph = pd.read_pickle(path="./dataset/temptest/iph_summary_validation.pkl")

    # 实际上,X里面的内容都是一样的,所以这里统一使用pw的X来作为所有的X
    # 但是标签是不一样的,所以需要每个都要具体定义
    X_train = np.asarray(list(df_train_pw['X'].values))
    X_validation = np.asarray(list(df_validation_pw['X'].values))

    # tags
    y_train_pw = np.asarray(list(df_train_pw['y'].values))
    y_validation_pw = np.asarray(list(df_validation_pw['y'].values))

    y_train_pph = np.asarray(list(df_train_pph['y'].values))
    y_validation_pph = np.asarray(list(df_validation_pph['y'].values))

    y_train_iph = np.asarray(list(df_train_iph['y'].values))
    y_validation_iph = np.asarray(list(df_validation_iph['y'].values))

    # length每一行序列的长度
    # 因为都一样,所以统一使用pw的
    len_train = np.asarray(list(df_train_pw['sentence_len'].values))
    len_validation = np.asarray(list(df_validation_pw['sentence_len'].values))
    print("len_train:", len_train.shape)
    print("len_validation:", len_validation.shape)

    # X_train = [X_train_pw, X_train_pph, X_train_iph]
    y_train = [y_train_pw, y_train_pph, y_train_iph]
    # X_validation = [X_validation_pw, X_validation_pph, X_validation_iph]
    y_validation = [y_validation_pw, y_validation_pph, y_validation_iph]

    # print("X_train_pw:\n",X_train_pw);      print(X_train_pw.shape)
    # print("X_train_pph:\n", X_train_pph);   print(X_train_pph.shape)
    # print("X_train_iph:\n", X_train_iph);   print(X_train_iph.shape)

    # print("y_train_pw:\n", y_train_pw);
    # print(y_train_pw.shape)
    # print("y_train_pph:\n", y_train_pph);
    # print(y_train_pph.shape)
    # print("y_train_iph:\n", y_train_iph);
    # print(y_train_iph.shape)

    model = Alignment_Seq2Seq()
    model.fit(X_train, y_train, len_train, X_validation, y_validation, len_validation, "test", False)