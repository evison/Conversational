"""
data_loader
"""

import numpy as np
import pickle
import random
import pandas as pd


class DataLoader:

    def __init__(self, args):
        self.args = args
        self.users_all = dict()
        self.user_number = 0
        self.w2v_dim = self.args.embed_dim
        self.input_mask_mode = "sentence"
        self.use_pretrained = self.args.use_pretrained
        self.train_batch_id = 0
        self.test_batch_id = 0

        self.base_path = self.args.base_path
        self.category = self.args.category

        path = self.base_path + self.category
        train_data_path = path + 'train_dict'
        self.train_data = pickle.load(open(train_data_path, 'rb'))
        test_data_path = path + 'test_dict'
        self.test_data = pickle.load(open(test_data_path, 'rb'))
        item_id_path = path + 'item_id_dict'
        self.items = list(pickle.load(open(item_id_path, 'rb')).values())
        print('item number: '+str(len(self.items)))

        feature_id_path = path + 'feature_id_dict'
        self.id_feature_dict = {v:k for k,v in pickle.load(open(feature_id_path, 'rb')).items()}

        opinion_id_path = path + 'opinion_id_dict'
        self.id_opinion_dict = {v: k for k, v in pickle.load(open(opinion_id_path, 'rb')).items()}
        word_id_path = path + 'word_id_dict'
        self.word_id_dict = pickle.load(open(word_id_path, 'rb'))


        item_description_dict_path = path + 'item_description_dict'
        self.item_description_dict = pickle.load(open(item_description_dict_path, 'rb'))
        item_category_dict_path = path + 'item_category_dict'
        self.item_category_dict = pickle.load(open(item_category_dict_path, 'rb'))

        # build for model testing
        self.item_candidates = []
        for k, v in self.test_data.items():
            item = int(k.split('@')[1])
            if item not in self.item_candidates:
                self.item_candidates.append(item)
        if len(self.item_candidates) > 100:
            self.item_candidates = random.sample(self.item_candidates, 100)
        self.grund_truth = dict()

        for k, v in self.test_data.items():
            user = int(k.split('@')[0])
            item = int(k.split('@')[1])
            if item in self.item_candidates:
                if user not in self.grund_truth.keys():
                    self.grund_truth[user] = [item]
                else:
                    self.grund_truth[user].append(item)

        self.question_cadidates = list(self.id_feature_dict.values())[:100]

    def make_train_and_test_set(self):
        train_raw = self.get_train_raw_data()
        test_raw = self.get_test_raw_data()
        self.train_sample_num = len(train_raw)
        self.test_sample_num = len(test_raw)
        self.max_description_word_length, self.max_description_sentence_length, \
        self.max_answer_word_length, self.max_answer_sentence_length = self.get_max_seq_length(train_raw, test_raw)
        self.max_description_word_length = 128

        print("max_description_word_length:", self.max_description_word_length)
        print("max_description_sentence_length:", self.max_description_sentence_length)
        print("max_answer_word_length:", self.max_answer_word_length)
        print("max_answer_sentence_length:", self.max_answer_sentence_length)
        print("train sample number:", self.train_sample_num)
        print("test sample number:", self.test_sample_num)
        print('test item number:%d' % (len(self.item_candidates)))

        self.train_users, self.train_answers, self.train_pos_descriptions, self.train_neg_descriptions, self.train_pos_questions, \
        self.train_neg_questions, self.train_answer_masks, self.train_pos_descriptions_masks, self.train_neg_descriptions_masks = self.process_train_input(train_raw)

        self.test_users, self.test_answers, self.test_pos_descriptions, self.test_pos_questions, \
        self.test_pos_descriptions_masks = self.process_test_input(test_raw)

    def get_max_seq_length(self, *datasets):

        max_description_word_length, max_description_sentence_length,\
        max_answer_word_length, max_answer_sentence_length = 0, 0, 0, 0

        def count_punctuation(facts):
            return len(list(filter(lambda x: x == ".", facts)))

        for dataset in datasets:
            for d in dataset:
                max_description_word_length = max(max_description_word_length, len(d['pos_des'].split('-')))
                max_description_sentence_length = max(max_description_sentence_length, count_punctuation(d['pos_des']))
                max_answer_word_length = max(max_answer_word_length, len(' '.join(d['answer']).split()))
                max_answer_sentence_length = max(max_answer_sentence_length, count_punctuation(d['answer']))

        return max_description_word_length, max_description_sentence_length,\
        max_answer_word_length, max_answer_sentence_length

    def get_all_description(self):
        all_d = []
        all_d_mask = []
        self.items = self.items[:10]
        for item in self.items:
            if int(item) in self.item_category_dict.keys() and int(item) in self.item_description_dict.keys():
                item_description_and_category = self.item_category_dict[int(item)].split('||')
                pos_product_description = '-'.join(item_description_and_category[0].split('-')[:10])
                pos_review_decription = '-'.join(self.item_description_dict[int(item)].split('-')[:100])
                d = pos_product_description + '-' + pos_review_decription
                pos_des = d.lower().split('-')
                pos_des = [self.word_id_dict[w] for w in pos_des if w in self.word_id_dict.keys()]
                pos_des_pad = self.pad_input(pos_des, self.max_description_word_length, [0])
                all_d.append(pos_des_pad)
                pos_mask = [index for index, w in enumerate(pos_des) if w == self.word_id_dict['.']]
                pos_mask = self.pad_input(pos_mask, self.max_description_sentence_length, [0])
                all_d_mask.append(pos_mask)
        return all_d, all_d_mask

    def get_train_raw_data(self):
        data = self.train_data.items()
        tasks = []


        for user_item, feature_opinion in list(data)[:100]:

            # Item description, category and the feature-opinion pairs.
            # 0. category
            # 1. caetgory + (feature1, opinion1)
            # 2. caetgory + (feature1, opinion1) + (feature2, opinion2)
            # ...
            task = {"user_item":"", "pos_des": "", "neg_des": "", "answer": "", "pos_ques": "", "neg_ques": ""}
            pos_item = user_item.split('@')[1]
            fo_pairs = [i.split('|')[:2] for i in feature_opinion.split(':')]
            # pos
            if int(pos_item) in self.item_category_dict.keys() and int(pos_item) in self.item_description_dict.keys():
                item_description_and_category = self.item_category_dict[int(pos_item)].split('||')
                category = ' '.join(set(item_description_and_category[1].split('-')))
                an = category
                task["user_item"] = user_item
                user = user_item.split('@')[0]
                if user not in self.users_all.keys():
                    self.users_all[user] = self.user_number
                    self.user_number += 1

                task["answer"] = an
                pos_product_description = '-'.join(item_description_and_category[0].split('-')[:10])
                pos_review_decription = '-'.join(self.item_description_dict[int(pos_item)].split('-')[:100])
                task["pos_des"] = pos_product_description + '-' + pos_review_decription
                task["pos_ques"] = self.id_feature_dict[int(fo_pairs[0][0])]

                neg_item = random.choice(self.items)
                while (neg_item == pos_item or int(neg_item) not in self.item_category_dict.keys() or int(neg_item) not in self.item_description_dict.keys()):
                    neg_item = random.choice(self.items)
                item_description_and_category = self.item_category_dict[int(neg_item)].split('||')

                neg_product_description = '-'.join(item_description_and_category[0].split('-')[:10])
                neg_review_decription = '-'.join(self.item_description_dict[int(neg_item)].split('-')[:100])
                task["neg_des"] = neg_product_description + '-' + neg_review_decription

                neg_ques = random.choice(list(self.id_feature_dict.values()))
                if neg_ques == int(fo_pairs[0][0]):
                    neg_ques = random.choice(list(self.id_feature_dict.values()))
                task["neg_ques"] = neg_ques
                tasks.append(task.copy())


                for index in range(len(fo_pairs)):
                    if index + 1 < len(fo_pairs):
                        f = self.id_feature_dict[int(fo_pairs[index][0])]
                        o = self.id_opinion_dict[int(fo_pairs[index][1])]
                        task["user_item"] = user_item
                        task["pos_des"] = pos_product_description + '-' + pos_review_decription
                        task["neg_des"] = neg_product_description + '-' + neg_review_decription
                        an += ' . ' + f + ' ' + o
                        task["answer"] = an
                        task["pos_ques"] = self.id_feature_dict[int(fo_pairs[index+1][0])]
                        neg_ques = random.choice(list(self.id_feature_dict.values()))
                        if neg_ques == int(fo_pairs[index+1][0]):
                            neg_ques = random.choice(list(self.id_feature_dict.values()))
                        task["neg_ques"] = neg_ques
                        tasks.append(task.copy())
        return tasks

    def get_test_raw_data(self):
        data = self.test_data.items()
        tasks = []
        output_search_result_index = []
        output_question_result_index = []

        '''
        item description (changing), item description mask, answer including n round conversation, question at n+1 round
        --> search task at round n
        item description, item description mask, answer including n round conversation, question at n+1 round (changing)
        --> question task at round n
        '''
        if self.args.evaluate == 'search':
            for user_item, feature_opinion in list(data)[:100]:
                task = {"user_item": "", "pos_des": "", "neg_des": "", "answer": "", "pos_ques": "", "neg_ques": ""}
                user = user_item.split('@')[0]
                #item = user_item.split('@')[1]
                fo_pairs = [i.split('|')[:2] for i in feature_opinion.split(':')]
                conversation = ''
                for i in range(self.args.search_with_conversation_number):
                    conversation += ' . ' + self.id_feature_dict[int(fo_pairs[i][0])] + ' ' + \
                                    self.id_opinion_dict[int(fo_pairs[i][1])]

                if int(user) in self.grund_truth.keys():
                    user = user_item.split('@')[0]
                    if user not in self.users_all.keys():
                        self.users_all[user] = self.user_number
                        self.user_number += 1

                    for pos_item in self.item_candidates:
                        if int(pos_item) in self.item_category_dict.keys() and int(pos_item) in self.item_description_dict.keys():
                            item_description_and_category = self.item_category_dict[int(pos_item)].split('||')
                            category = ' '.join(set(item_description_and_category[1].split('-')))
                            an = category
                            task["user_item"] = user_item
                            task["answer"] = an + conversation
                            pos_product_description = '-'.join(item_description_and_category[0].split('-')[:10])
                            pos_review_decription = '-'.join(self.item_description_dict[int(pos_item)].split('-')[:100])
                            task["pos_des"] = pos_product_description + '-' + pos_review_decription
                            task["pos_ques"] = ''
                            if pos_item in self.grund_truth[int(user)]:
                                tmp = [user, pos_item, 1]
                            else:
                                tmp = [user, pos_item, 0]
                            output_search_result_index.append(tmp)
                            tasks.append(task.copy())
            t = pd.DataFrame(output_search_result_index)
            t.to_csv(self.base_path + self.category+'output_'+self.args.evaluate+'_result_index', index=False, header=None)
            return tasks
        else:
            # predict the first aspect when there is no conversation.
            for user_item, feature_opinion in list(data)[:100]:
                task = {"user_item": "", "pos_des": "", "neg_des": "", "answer": "", "pos_ques": "", "neg_ques": ""}
                user = user_item.split('@')[0]
                if user not in self.users_all.keys():
                    self.users_all[user] = self.user_number
                    self.user_number += 1
                pos_item = user_item.split('@')[1]
                fo_pairs = [i.split('|')[:2] for i in feature_opinion.split(':')]
                conversation = ''
                if len(fo_pairs) > self.args.prediction_with_conversation_number:
                    for i in range(self.args.prediction_with_conversation_number):
                        conversation += ' . ' + self.id_feature_dict[int(fo_pairs[i][0])] + ' ' + \
                                        self.id_opinion_dict[int(fo_pairs[i][1])]

                    if int(pos_item) in self.item_category_dict.keys() and int(
                            pos_item) in self.item_description_dict.keys():
                        item_description_and_category = self.item_category_dict[int(pos_item)].split('||')
                        category = ' '.join(set(item_description_and_category[1].split('-')))
                        an = category
                        pos_product_description = ' '.join(item_description_and_category[0].split('-')[:10])
                        pos_review_decription = self.item_description_dict[int(pos_item)]

                        for pos_ques in self.question_cadidates:
                            task["user_item"] = user_item
                            task["answer"] = an + conversation
                            task["pos_des"] = pos_product_description + ' ' + pos_review_decription
                            task["pos_ques"] = pos_ques
                            if pos_ques == self.id_feature_dict[int(fo_pairs[self.args.prediction_with_conversation_number][0])]:
                                tmp = [user, pos_item, pos_ques, 1]
                            else:
                                tmp = [user, pos_item, pos_ques, 0]
                            output_question_result_index.append(tmp)
                            tasks.append(task.copy())

            t = pd.DataFrame(output_question_result_index)
            t.to_csv(self.base_path + self.category+'output_'+self.args.evaluate+'_result_index', index=False, header=None)
            return tasks

    def get_norm(self, x):
        x = np.array(x)
        return np.sum(x * x)

    def process_train_input(self, data_raw):
        users = []
        answers = []
        pos_descriptions = []
        neg_descriptions = []
        pos_questions = []
        neg_questions = []

        pos_descriptions_masks = []
        neg_descriptions_masks = []
        answer_masks = []

        for x in data_raw:
            user = x["user_item"].split('@')[0]
            users.append(self.users_all[user])

            pos_des = x["pos_des"].lower().split('-')
            pos_des = [self.word_id_dict[w] for w in pos_des if w in self.word_id_dict.keys()]

            neg_des = x["neg_des"].lower().split('-')
            neg_des = [self.word_id_dict[w] for w in neg_des if w in self.word_id_dict.keys()]

            an = x["answer"].lower().split(' ')
            an = [self.word_id_dict[w] for w in an if w in self.word_id_dict.keys()]

            pos_des_pad = self.pad_input(pos_des, self.max_description_word_length, [0])
            pos_descriptions.append(pos_des_pad)

            neg_des_pad = self.pad_input(neg_des, self.max_description_word_length, [0])
            neg_descriptions.append(neg_des_pad)

            an_pad = self.pad_input(an, self.max_answer_word_length, [0])
            answers.append(an_pad)

            pos_questions.append(self.word_id_dict[x["pos_ques"]])
            neg_questions.append(self.word_id_dict[x["neg_ques"]])


            if self.input_mask_mode == 'word':
                pos_mask_tmp = [index for index, w in enumerate(pos_des)]
                pos_mask = self.pad_input(pos_mask_tmp, self.max_description_word_length, [0])
                pos_descriptions_masks.append(pos_mask)
                neg_mask_tmp = [index for index, w in enumerate(neg_des)]
                neg_mask = self.pad_input(neg_mask_tmp, self.max_description_word_length, [0])
                neg_descriptions_masks.append(neg_mask)
                answer_mask_tmp = [index for index, w in enumerate(an)]
                answer_mask = self.pad_input(answer_mask_tmp, self.max_answer_word_length, [0])
                answer_masks.append(answer_mask)

            elif self.input_mask_mode == 'sentence':
                pos_mask = [index for index, w in enumerate(pos_des) if w == self.word_id_dict['.']]
                pos_mask = self.pad_input(pos_mask, self.max_description_sentence_length, [0])
                pos_descriptions_masks.append(pos_mask)
                neg_mask = [index for index, w in enumerate(neg_des) if w == self.word_id_dict['.']]
                neg_mask = self.pad_input(neg_mask, self.max_description_sentence_length, [0])
                neg_descriptions_masks.append(neg_mask)
                answer_mask_tmp = [index for index, w in enumerate(an)]
                answer_mask = self.pad_input(answer_mask_tmp, self.max_answer_word_length, [0])
                answer_masks.append(answer_mask)
                #answer_mask = [index for index, w in enumerate(an) if w == self.word_id_dict['.']]
                #answer_mask = self.pad_input(answer_mask, self.max_answer_sentence_length, [0])
                #answer_masks.append(answer_mask)

            else:
                raise ValueError("input_mask_mode is only available (word, sentence)")

        return (np.array(users, dtype=np.int32).tolist(),
                np.array(answers, dtype=np.int32).tolist(),
                np.array(pos_descriptions, dtype=np.int32).tolist(),
                np.array(neg_descriptions, dtype=np.int32).tolist(),
                np.array(pos_questions, dtype=np.int32).tolist(),
                np.array(neg_questions, dtype=np.int32).tolist(),
                np.array(answer_masks, dtype=np.int32).tolist(),
                np.array(pos_descriptions_masks, dtype=np.int32).tolist(),
                np.array(neg_descriptions_masks, dtype=np.int32).tolist())

    def process_test_input(self, data_raw):
        users = []
        answers = []
        descriptions = []
        questions = []
        descriptions_masks = []
        for x in data_raw:
            user = x["user_item"].split('@')[0]
            users.append(self.users_all[user])
            pos_des = x["pos_des"].lower().split('-')
            pos_des = [self.word_id_dict[w] for w in pos_des if w in self.word_id_dict.keys()]
            an = x["answer"].lower().split(' ')
            an = [self.word_id_dict[w] for w in an if w in self.word_id_dict.keys()]
            des_pad = self.pad_input(pos_des, self.max_description_word_length, [0])
            descriptions.append(des_pad)
            an_pad = self.pad_input(an, self.max_answer_word_length, [0])
            answers.append(an_pad)
            if self.args.evaluate == 'search':
                questions.append(0)
            else:
                questions.append(self.word_id_dict[x["pos_ques"]])

            if self.input_mask_mode == 'word':
                descriptions_masks.append(np.array([index for index, w in enumerate(pos_des)], dtype=np.int32))
            elif self.input_mask_mode == 'sentence':
                pos_mask = [index for index, w in enumerate(pos_des) if w == self.word_id_dict['.']]
                pos_mask = self.pad_input(pos_mask, self.max_description_sentence_length, [0])
                descriptions_masks.append(pos_mask)
            else:
                raise ValueError("input_mask_mode is only available (word, sentence)")

        return (np.array(users, dtype=np.int32).tolist(),
                np.array(answers, dtype=np.int32).tolist(),
                np.array(descriptions, dtype=np.int32).tolist(),
                np.array(questions, dtype=np.int32).tolist(),
                np.array(descriptions_masks, dtype=np.int32).tolist())

    def pad_input(self, input_, size, pad_item):
        if size > len(input_):
            return input_ + pad_item * (size - len(input_))
        else:
            return input_[:size]

    def get_train_batch_data(self, batch_size):
        l = len(self.train_answers)
        if self.train_batch_id + batch_size > l:
            batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[:self.train_batch_id + batch_size - l]
            batch_train_answers = self.train_answers[self.train_batch_id:] + self.train_answers[:self.train_batch_id + batch_size - l]
            batch_train_pos_descriptions = self.train_pos_descriptions[self.train_batch_id:] + self.train_pos_descriptions[:self.train_batch_id + batch_size - l]
            batch_train_neg_descriptions = self.train_neg_descriptions[self.train_batch_id:] + self.train_neg_descriptions[:self.train_batch_id + batch_size - l]

            batch_train_pos_questions = self.train_pos_questions[self.train_batch_id:] + self.train_pos_questions[:self.train_batch_id + batch_size - l]
            batch_train_neg_questions = self.train_neg_questions[self.train_batch_id:] + self.train_neg_questions[:self.train_batch_id + batch_size - l]

            batch_train_answer_masks = self.train_answer_masks[self.train_batch_id:] + self.train_answer_masks[:self.train_batch_id + batch_size - l]
            batch_train_pos_descriptions_masks = self.train_pos_descriptions_masks[self.train_batch_id:] + self.train_pos_descriptions_masks[:self.train_batch_id + batch_size - l]
            batch_train_neg_descriptions_masks = self.train_neg_descriptions_masks[self.train_batch_id:] + self.train_neg_descriptions_masks[:self.train_batch_id + batch_size - l]

            self.train_batch_id = self.train_batch_id + batch_size - l

        else:
            batch_train_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_answers = self.train_answers[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_descriptions = self.train_pos_descriptions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_descriptions = self.train_neg_descriptions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_questions = self.train_pos_questions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_questions = self.train_neg_questions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_answer_masks = self.train_answer_masks[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_descriptions_masks = self.train_pos_descriptions_masks[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_descriptions_masks = self.train_neg_descriptions_masks[self.train_batch_id:self.train_batch_id + batch_size]

            self.train_batch_id = self.train_batch_id + batch_size

        return [batch_train_answers, batch_train_pos_descriptions, batch_train_neg_descriptions, \
               batch_train_pos_questions, batch_train_neg_questions, batch_train_answer_masks, \
               batch_train_pos_descriptions_masks, batch_train_neg_descriptions_masks, batch_train_users]

    def get_test_batch_data(self, batch_size):
        l = len(self.test_answers)
        if self.test_batch_id + batch_size > l:
            batch_test_users = self.test_users[self.test_batch_id:] + self.test_users[:self.test_batch_id + batch_size - l]
            batch_test_answers = self.test_answers[self.test_batch_id:] + self.test_answers[:self.test_batch_id + batch_size - l]
            batch_test_pos_descriptions = self.test_pos_descriptions[self.test_batch_id:] + self.test_pos_descriptions[:self.test_batch_id + batch_size - l]
            batch_test_pos_questions = self.test_pos_questions[self.test_batch_id:] + self.test_pos_questions[:self.test_batch_id + batch_size - l]
            batch_test_pos_descriptions_masks = self.test_pos_descriptions_masks[self.test_batch_id:] + self.test_pos_descriptions_masks[:self.test_batch_id + batch_size - l]

            self.test_batch_id = self.test_batch_id + batch_size - l

        else:
            batch_test_users = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_answers = self.test_answers[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_descriptions = self.test_pos_descriptions[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_questions = self.test_pos_questions[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_descriptions_masks = self.test_pos_descriptions_masks[self.test_batch_id:self.test_batch_id + batch_size]

            self.test_batch_id = self.test_batch_id + batch_size

        return [batch_test_answers, batch_test_pos_descriptions, batch_test_pos_questions, batch_test_pos_descriptions_masks, batch_test_users]