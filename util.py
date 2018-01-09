c1_iph, c2_iph= sess.run(
                        fetches=[self.accuracy_class_1_iph,
                                 self.accuracy_class_2_iph],
                        feed_dict={
                            self.X_p: X_train_iph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.y_p: y_train_iph[i * self.batch_size:(i + 1) * self.batch_size],
                            self.class1_p: np.full(shape=(self.batch_size * self.max_sentence_size,), fill_value=1),
                            self.class2_p: np.full(shape=(self.batch_size * self.max_sentence_size,), fill_value=2)
                        }
                    )
                    c1_accus_iph.append(c1_iph);      c2_accus_iph.append(c2_iph)