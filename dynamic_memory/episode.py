import tensorflow as tf



class Episode:
    """Episode class is used update memory in in Episodic Memory Module"""

    def __init__(self, num_units, reg_scale=0.001, reuse=False):
        self.gate = AttentionGate(hidden_size=num_units, reg_scale=reg_scale, reuse=reuse)
        self.rnn = tf.contrib.rnn.GRUCell(num_units)

    def update(self, c, m_t, q_t):
        """Update memory with attention mechanism

        * Args:
            c : encoded raw text and stacked by each sentence
                shape: fact_count x [batch_size, num_units]
            m_t : previous memory
                shape: [num_units, batch_size]
            q_t : encoded question last state
                shape: [num_units, batch_size]

        * Returns:
            h : updated memory
        """
        h = tf.zeros_like(c[0])
        init_w = self.gate.score(tf.transpose(c[0], name="init_shape"), m_t, q_t)
        w = tf.zeros_like(init_w)
        with tf.variable_scope('memory-update') as scope:
            for fact in c:
                g = self.gate.score(tf.transpose(fact, name="c"), m_t, q_t)
                #h = g * self.rnn(fact, h, scope="episode_rnn")[0] + (1 - g) * h
                w = tf.add(g, w)
                h = tf.add(g * fact, h)
                scope.reuse_variables()
        return tf.div(h, w)


class AttentionGate:
    """AttentionGate class is simple two-layer feed forward neural network with Score function."""

    def __init__(self, hidden_size=4, reg_scale=0.001,reuse=False):
        with tf.variable_scope('attention_weight', reuse=reuse) as scope:
            self.w1 = tf.get_variable(
                "w1", [hidden_size, 3 * hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
            self.b1 = tf.get_variable("b1", [hidden_size, 1])
            self.w2 = tf.get_variable(
                "w2", [1, hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
            self.b2 = tf.get_variable("b2", [1, 1])

    def score(self, c_t, m_t, q_t):
        """For captures a variety of similarities between input(c), memory(m) and question(q)

        * Args:
            c_t : transpose of one fact (encoded sentence's last state)
                  shape: [num_units, batch_size]
            m_t : transpose of previous memory
                  shape: [num_units, batch_size]
            q_t : transpose of encoded question
                  shape: [num_units, batch_size]

        * Returns:
            gate score
            shape: [batch_size, 1]
        """

        with tf.variable_scope('attention_gate'):
            #z = tf.concat([c_t, m_t, q_t, c_t*q_t, c_t*m_t, (c_t-q_t)**2, (c_t-m_t)**2], 0)
            z = tf.concat([c_t, m_t, q_t], 0)
            o1 = tf.nn.tanh(tf.matmul(self.w1, z) + self.b1)
            o2 = tf.nn.sigmoid(tf.matmul(self.w2, o1) + self.b2)
            return tf.transpose(o2)
