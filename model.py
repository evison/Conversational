from __future__ import print_function
import tensorflow as tf
import dynamic_memory
import pickle


class Model:
    def __init__(self):
        pass

    def build_graph_init(self, params):
        self.params = params
        self.base_path = params['base_path']
        self.category = params['category']
        path = self.base_path + self.category
        word_id_path = path + 'word_id_dict'
        self.word_id_dict = pickle.load(open(word_id_path, 'rb'))
        word_number = len(self.word_id_dict.items())

        self.user_placeholder = tf.placeholder(tf.int32, [None])
        self.all_description_placeholder = tf.placeholder(tf.int32, [params['item_number'], params['max_description_word_length']])
        self.pos_description_placeholder = tf.placeholder(tf.int32, [None, params['max_description_word_length']])
        self.neg_description_placeholder = tf.placeholder(tf.int32, [None, params['max_description_word_length']])
        self.answer_placeholder = tf.placeholder(tf.int32, [None, params['max_answer_word_length']])
        self.pos_question_placeholder = tf.placeholder(tf.int32, [None])
        self.neg_question_placeholder = tf.placeholder(tf.int32, [None])

        self.all_descriptions_mask_placeholder = tf.placeholder(tf.int32,[params['item_number'], params['max_description_sentence_length']])
        self.pos_descriptions_mask_placeholder = tf.placeholder(tf.int32,[None, params['max_description_sentence_length']])
        self.neg_descriptions_mask_placeholder = tf.placeholder(tf.int32,[None, params['max_description_sentence_length']])
        self.answer_mask_placeholder = tf.placeholder(tf.int32, [None, params['max_answer_word_length']])

        self.initializer = tf.random_uniform_initializer(minval=-1, maxval=1)
        self.word_embedding_matrix = tf.get_variable('word_embedding', [word_number, params['embed_dim']], initializer=self.initializer)
        self.user_embedding_matrix = tf.get_variable('user_embedding', [self.params['user_number'], params['embed_dim']], initializer=self.initializer)

        self.embedding_user = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_placeholder)
        self.embedding_all_description = tf.nn.embedding_lookup(self.word_embedding_matrix, self.all_description_placeholder)
        self.embedding_pos_description = tf.nn.embedding_lookup(self.word_embedding_matrix, self.pos_description_placeholder)
        self.embedding_neg_description = tf.nn.embedding_lookup(self.word_embedding_matrix, self.neg_description_placeholder)
        self.embedding_answer = tf.nn.embedding_lookup(self.word_embedding_matrix, self.answer_placeholder)
        self.embedding_pos_question = tf.nn.embedding_lookup(self.word_embedding_matrix, self.pos_question_placeholder)
        self.embedding_neg_question = tf.nn.embedding_lookup(self.word_embedding_matrix, self.neg_question_placeholder)


        self.dtype = tf.float32
        self.loss, self.train_op, self.predictions = None, None, None
        self.graph = dynamic_memory.Graph(self.params)

    def build_graph(self):
        self.loss = self.graph.build_loss(
                                     embedding_user=self.embedding_user,
                                     embedding_all_description=self.embedding_all_description,
                                     embedding_pos_description = self.embedding_pos_description,
                                     embedding_neg_description = self.embedding_neg_description,
                                     embedding_answer = self.embedding_answer,
                                     all_description_mask=self.all_descriptions_mask_placeholder,
                                     pos_description_mask = self.pos_descriptions_mask_placeholder,
                                     neg_description_mask = self.neg_descriptions_mask_placeholder,
                                     embedding_pos_question = self.embedding_pos_question,
                                     embedding_neg_question = self.embedding_neg_question
                                     )
        # _build_prediction should after build_loss
        self._build_prediction()
        self._build_optimizer()

    def _build_prediction(self):
        self.search_predictions = self.graph.build_search_prediction(
                                                       embedding_user=self.embedding_user,
                                                       embedding_description = self.embedding_pos_description,
                                                       embedding_answer = self.embedding_answer,
                                                       description_mask = self.pos_descriptions_mask_placeholder)

        self.question_predictions = self.graph.build_question_prediction(
                                        embedding_user=self.embedding_user,
                                        all_embedding_description=self.embedding_all_description,
                                                              embedding_description=self.embedding_pos_description,
                                                              embedding_answer=self.embedding_answer,
                                             all_description_mask=self.all_descriptions_mask_placeholder,
                                                              description_mask=self.pos_descriptions_mask_placeholder,
                                                              embedding_question=self.embedding_pos_question)


    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=self.params['optimizer'],
            learning_rate=self.params['learning_rate'],
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")

