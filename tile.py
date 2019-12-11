import tensorflow as tf
temp = tf.tile([[[1,3],[2,1],[3,5]],[[4,1],[2,1],[5,4]]],[3,1,1])
temp2 = tf.reshape(tf.tile([[[1,3],[2,1],[3,5]],[[4,1],[2,1],[5,4]]],[1,3,1]),[-1, 3, 2])

with tf.Session() as sess:
    print(sess.run(temp))
    print(sess.run(temp2))
