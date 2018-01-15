# logits
logits_iph = tf.matmul(h_iph, w_iph) + b_iph  # shape of logits:[batch_size*max_time, 3]
logits_normal_iph = tf.reshape(  # logits in an normal way:[batch_size,max_time_stpes,3]
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
self.loss_iph = tf.losses.softmax_cross_entropy(
    labels=y_p_iph_masked,
    logits=logits_iph_masked
)
