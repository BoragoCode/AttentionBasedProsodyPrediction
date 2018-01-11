import tensorflow as tf
import numpy as np

a=tf.constant(value=[[4,2,3],[2,2,2],[3,3,3]])
b=tf.constant(value=[[1,2,3],[2,2,2],[3,3,3]])
d=[a,b]


#a_col=a[:,0]
#a_col=tf.reshape(tensor=a_col,shape=(-1,1))
#sum_d=sum(d)

#d_t=tf.zeros(shape=(2,3,3))
#d_t[0,:,:]=a
#d_t[1,:,:]=b

d_t=tf.concat(values=d,axis=0)



#c=tf.multiply(x=a_col,y=b)

with tf.Session() as sess:
    print("a\n",sess.run(a))
    #print("a\n", sess.run(a_col))

    print("b\n", sess.run(b))
    print("d_t\n", sess.run(d_t))