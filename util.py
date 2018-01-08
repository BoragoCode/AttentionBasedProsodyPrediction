'''
            #---------------------------------/hierarchy:PW/-------------------------------------------#
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pw = tf.nn.embedding_lookup(params=embeddings, ids=self.X_p_pw, name="embeded_input_pw")
            #encoder cells
            #forward part
            en_lstm_forward1_pw=rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            #en_lstm_forward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            #en_lstm_forward=rnn.MultiRNNCell(cells=[en_lstm_forward1,en_lstm_forward2])

            #backward part
            en_lstm_backward1_pw=rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            #en_lstm_backward2=rnn.BasicLSTMCell(num_units=self.hidden_units_num2)
            #en_lstm_backward=rnn.MultiRNNCell(cells=[en_lstm_backward1,en_lstm_backward2])

            #decoder cells
            de_lstm_pw = rnn.BasicLSTMCell(num_units=self.hidden_units_num)

            #encode
            #encoder_outputs_pw, encoder_states_pw=self.encoder(
            #        cell_forward=en_lstm_forward1_pw,
            #        cell_backward=en_lstm_backward1_pw,
            #        inputs=inputs_pw
            #)

            # shape of h is [batch*time_steps,hidden_units]
            # h_pw=self.decoder(cell=de_lstm_pw,initial_state=states_forward_pw,inputs=encoder_outputs_pw)

            #encode
            en_outputs_pw, en_states_pw = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=en_lstm_forward1_pw,
                cell_bw=en_lstm_backward1_pw,
                inputs=inputs_pw,
                dtype=tf.float32
            )
            #print("shape of states:", states_pw)
            outputs_forward_pw = en_outputs_pw[0]  # shape of h is [batch_size, max_time, cell_fw.output_size]
            outputs_backward_pw = en_outputs_pw[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]
            # shape of h is [batch_size, max_time, cell_fw.output_size*2]
            encoder_outputs_pw = tf.concat(values=[outputs_forward_pw, outputs_backward_pw], axis=2)
            states_forward_pw = en_states_pw[0]

            #decode
            de_outputs_pw, de_states_pw = tf.nn.dynamic_rnn(
                cell=de_lstm_pw,
                inputs=encoder_outputs_pw,
                initial_state=states_forward_pw
            )

            # outputs      #[batch_size,time_steps,hidden_size*2]
            h_pw = tf.reshape(tensor=de_outputs_pw, shape=(-1, self.hidden_units_num))
            #return decoder_outputs

            #fully connect layer
            w_pw=tf.Variable(
                initial_value=tf.random_normal(shape=(self.hidden_units_num2,self.class_num)),
                name="weights_pw"
            )
            b_pw=tf.Variable(
                initial_value=tf.random_normal(shape=(self.class_num,)),
                name="bias_pw"
            )
            logits_pw=tf.matmul(h_pw,w_pw)+b_pw          #shape of logits:[batch_size*max_time, 5]

            #prediction
            # shape of pred[batch_size*max_time, 1]
            pred_pw=tf.cast(tf.argmax(logits_pw, 1), tf.int32,name="pred_pw")

            # pred in an normal way,shape is [batch_size, max_time]
            pred_normal_pw=tf.reshape(
                tensor=pred_pw,
                shape=(-1,self.max_sentence_size),
                name="pred_normal"
            )

            # one-hot the pred_normal:[batch_size, max_time,class_num]
            pred_normal_one_hot_pw=tf.one_hot(
                indices=pred_normal_pw,
                depth=self.class_num
            )

            #correct_prediction
            correct_prediction_pw = tf.equal(pred_pw, tf.reshape(self.y_p_pw, [-1]))
            #accracy
            self.accuracy_pw=tf.reduce_mean(
                input_tensor=tf.cast(x=correct_prediction_pw,dtype=tf.float32),
                name="accuracy_pw"
            )
            # class #1
            # class1=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=1)
            basic_class_1_pw = tf.cast(tf.equal(self.class1_p, tf.reshape(self.y_p_pw, [-1])), dtype=tf.int32)
            pred_class_1_pw = tf.cast(tf.equal(self.class1_p, tf.reshape(pred_pw, [-1])), dtype=tf.int32)
            correct_class_1_pw = tf.bitwise.bitwise_and(basic_class_1_pw, pred_class_1_pw)  # #1 prediction

            self.accuracy_class_1_pw = tf.divide(
                x=tf.reduce_sum(correct_class_1_pw),
                y=tf.reduce_sum(basic_class_1_pw),
                name="accuracy_class_1_pw"
            )

            # class #2
            # class2=np.full(shape=(self.batch_size*self.max_sentence_size,),fill_value=2)
            basic_class_2_pw = tf.cast(tf.equal(self.class2_p, tf.reshape(self.y_p_pw, [-1])), dtype=tf.int32)
            pred_class_2_pw = tf.cast(tf.equal(self.class2_p, tf.reshape(pred_pw, [-1])), dtype=tf.int32)
            correct_class_2_pw = tf.bitwise.bitwise_and(basic_class_2_pw, pred_class_2_pw)  # #2 prediction
            self.accuracy_class_2_pw = tf.divide(
                x=tf.reduce_sum(correct_class_2_pw),
                y=tf.reduce_sum(basic_class_2_pw),
                name="accuracy_class_2_pw"
            )

            #loss
            self.loss_pw=tf.losses.sparse_softmax_cross_entropy(
                labels=tf.reshape(self.y_p_pw,shape=[-1]),
                logits=logits_pw
            )
            #--------------------------------------------------------------------------------------------#
            '''