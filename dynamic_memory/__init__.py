import tensorflow as tf
from .encoder import Encoder
from .episode import Episode



class Graph:

    def __init__(self, params):
        self.params = params

    def build_loss(self,
                   embedding_user,
                   embedding_all_description,
                   embedding_pos_description,
                   embedding_neg_description,
                   embedding_answer,
                   all_description_mask,
                   pos_description_mask,
                   neg_description_mask,
                   embedding_pos_question,
                   embedding_neg_question):
        self.all = self.params['item_number']

        facts, answer = self._build_input_module(embedding_pos_description, pos_description_mask, embedding_answer, reuse=False)
        pos_search_memory, _ = self._build_episodic_memory(facts, answer, reuse=False)
        pos_search = self._build_search_decoder(embedding_user, pos_search_memory, answer, reuse=False)
        facts, answer = self._build_input_module(embedding_neg_description, neg_description_mask, embedding_answer, reuse=True)
        neg_search_memory, _ = self._build_episodic_memory(facts, answer, reuse=True)
        neg_search = self._build_search_decoder(embedding_user, neg_search_memory, answer, reuse=True)

        scat_embedding_all_description = tf.tile(embedding_all_description, [self.params['batch_size'], 1, 1])
        scat_all_description_mask = tf.tile(all_description_mask, [self.params['batch_size'], 1])
        scat_embedding_answer = tf.reshape(tf.tile(embedding_answer, [1, self.all, 1]),[-1, self.params['max_answer_word_length'], self.params['embed_dim']])

        scat_facts, scat_answer = self._build_input_module(scat_embedding_all_description, scat_all_description_mask, scat_embedding_answer, reuse=True)
        _, question_memory = self._build_episodic_memory(scat_facts, scat_answer, reuse=True)
        pos_question = self._build_question_decoder(embedding_user, question_memory, answer, embedding_pos_question, reuse=False)
        neg_question = self._build_question_decoder(embedding_user, question_memory, answer, embedding_neg_question, reuse=True)


        with tf.variable_scope('search_loss'):
            w = tf.get_variable("w", [self.params['num_units']/2, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b = tf.get_variable("b", [1, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            pos_s = tf.log_sigmoid(tf.matmul(pos_search, w) + b)
            neg_s = tf.log_sigmoid(-tf.matmul(neg_search, w) - b)
            search_loss = - pos_s - neg_s


        with tf.variable_scope('question_loss'):
            w = tf.get_variable("w", [self.params['num_units'], 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b = tf.get_variable("b", [1, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            pos_q = tf.log_sigmoid(tf.matmul(pos_question, w) + b)
            neg_q = tf.log_sigmoid(-tf.matmul(neg_question, w) - b)
            question_loss = - pos_q - neg_q

        reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.reduce_mean(tf.add(search_loss, question_loss))+reg_term
        return total_loss


    def _build_input_module(self, embedding_input, input_mask, embedding_question, reuse=False):
        encoder = Encoder(
            encoder_type=self.params['encoder_type'],
            num_layers=self.params['num_layers'],
            cell_type=self.params['cell_type'],
            num_units=self.params['num_units'],
            dropout=self.params['dropout'])

        # slice zeros padding
        input_length = tf.reduce_max(input_mask, axis=1)
        question_length = tf.reduce_sum(tf.to_int32(tf.not_equal(tf.reduce_max(embedding_question, axis=2),
                                                                 self.params['PAD_ID'])), axis=1)

        with tf.variable_scope("input-module") as scope:
            input_encoder_outputs, _ = encoder.build(embedding_input, input_length, scope="encoder", reuse=reuse)

            with tf.variable_scope("facts") as scope:
                batch_size = tf.shape(input_mask)[0]
                max_mask_length = tf.shape(input_mask)[1]

                def get_encoded_fact(i):
                    nonlocal input_mask

                    mask_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_mask[i], self.params['PAD_ID'])), axis=0)
                    input_mask = tf.boolean_mask(input_mask[i], tf.sequence_mask(mask_lengths, max_mask_length))

                    encoded_facts = tf.gather_nd(input_encoder_outputs[i], tf.reshape(input_mask, [-1, 1]))
                    padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, self.params['num_units']]))
                    return tf.concat([encoded_facts, padding], 0)

                facts_stacked = tf.map_fn(get_encoded_fact, tf.range(start=0, limit=batch_size), dtype=tf.float32)
                # max_input_mask_length x [batch_size, num_units]
                facts = tf.unstack(tf.transpose(facts_stacked, [1, 0, 2]), num=self.params['max_description_sentence_length'])
            
        with tf.variable_scope("input-module") as scope:
            scope.reuse_variables()
            _, question = encoder.build(embedding_question, question_length, scope="encoder", reuse=reuse)
        return facts, question[0]

    def _build_episodic_memory(self, facts, question, reuse=False):
        with tf.variable_scope('episodic-memory-module', reuse=reuse) as scope:
            memory = tf.identity(question)
            episode = Episode(self.params['num_units'], reg_scale=self.params['reg_scale'], reuse=reuse)
            rnn = tf.contrib.rnn.GRUCell(self.params['num_units'])
            updated_memory = episode.update(facts, tf.transpose(memory, name="m"), tf.transpose(question, name="q"))
            search_memory, _ = rnn(updated_memory, memory, scope="memory_rnn")
            scope.reuse_variables()
            updated_memory = episode.update(facts, tf.transpose(memory, name="m"), tf.transpose(question, name="q"))
            question_memory, _ = rnn(updated_memory, memory, scope="memory_rnn")
        return search_memory, question_memory

    def _build_search_decoder(self, embedding_user, last_memory, current_answer, reuse=False):
        with tf.variable_scope('search-module', reuse=reuse):
            w1 = tf.get_variable("w1", [self.params['embed_dim']+ 2 * self.params['num_units'], self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b1 = tf.get_variable("b1", [1, self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            w2 = tf.get_variable("w2", [self.params['num_units'],self.params['num_units']/2], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b2 = tf.get_variable("b2", [1, self.params['num_units']/2], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            z = tf.concat([current_answer, last_memory, embedding_user], 1)
            o1 = tf.nn.elu(tf.matmul(z, w1) + b1)
            o2 = tf.nn.elu(tf.matmul(o1, w2) + b2)
        # return $x_i$, which will be used to compute the probability of item i
        return o2

    def _build_question_decoder(self, embedding_user, last_memory, current_answer, next_question, reuse=False):
        last_memory_mean = tf.reduce_mean(tf.reshape(last_memory, [-1, self.all, self.params['num_units']]), 1)

        with tf.variable_scope('question-module', reuse=reuse):
            w1 = tf.get_variable("w1", [2 * self.params['num_units']+ 2*self.params['embed_dim'], 2*self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b1 = tf.get_variable("b1", [1, 2*self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            w2 = tf.get_variable("w2", [2*self.params['num_units'], self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b2 = tf.get_variable("b2", [1, self.params['num_units']], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))

            z = tf.concat([current_answer, last_memory_mean, next_question, embedding_user], 1)
            o1 = tf.nn.elu(tf.matmul(z, w1) + b1)
            o2 = tf.nn.elu(tf.matmul(o1, w2) + b2)
        # return $x_{k+1}$, which will be used to compute the probability of asking the next question (k+1)
        return o2

    def build_search_prediction(self, embedding_user, embedding_description, embedding_answer, description_mask):
        facts, answer = self._build_input_module(embedding_description, description_mask, embedding_answer, reuse=True)
        search_memory, question_memory = self._build_episodic_memory(facts, answer, reuse=True)
        search = self._build_search_decoder(embedding_user, search_memory, answer, reuse=True)
        with tf.variable_scope('search_loss', reuse=True):
            w = tf.get_variable("w", [self.params['num_units']/2, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b = tf.get_variable("b", [1, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            s = tf.log_sigmoid(tf.matmul(search, w) + b)
        return s

    def build_question_prediction(self, embedding_user, embedding_description, description_mask, all_embedding_description, embedding_answer, all_description_mask, embedding_question):
        _, answer = self._build_input_module(embedding_description, description_mask, embedding_answer, reuse=True)
        scat_embedding_all_description = tf.tile(all_embedding_description, [self.params['batch_size'], 1, 1])
        scat_all_description_mask = tf.tile(all_description_mask, [self.params['batch_size'], 1])
        scat_embedding_answer = tf.reshape(tf.tile(embedding_answer, [1, self.all, 1]),[-1, self.params['max_answer_word_length'],
                                            self.params['embed_dim']])
        scat_facts, scat_answer = self._build_input_module(scat_embedding_all_description, scat_all_description_mask, scat_embedding_answer, reuse=True)
        _, question_memory = self._build_episodic_memory(scat_facts, scat_answer, reuse=True)
        question = self._build_question_decoder(embedding_user, question_memory, answer, embedding_question, reuse=True)


        with tf.variable_scope('question_loss', reuse=True):
            w = tf.get_variable("w", [self.params['num_units'], 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            b = tf.get_variable("b", [1, 1], regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_scale']))
            q = tf.log_sigmoid(tf.matmul(question, w) + b)
        return q
