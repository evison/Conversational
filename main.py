#-- coding: utf-8 -*-
import argparse
import configparser
import tensorflow as tf
from data_loader import DataLoader
from model import Model
import pandas as pd
import numpy as np

class solver():

    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.data_loader = DataLoader(self.args)
        self.data_loader.make_train_and_test_set()
        self.train_sample_num = self.data_loader.train_sample_num
        self.test_sample_num = self.data_loader.test_sample_num
        self.all_d, self.all_d_mask = self.data_loader.get_all_description()

        self.params = {}
        for k, v in vars(args).items():
            self.params[k] = v
        self.params['max_description_word_length'] = self.data_loader.max_description_word_length
        self.params['max_description_sentence_length'] = self.data_loader.max_description_sentence_length
        self.params['max_answer_word_length'] = self.data_loader.max_answer_word_length
        self.params['max_answer_sentence_length'] = self.data_loader.max_answer_sentence_length
        self.params['item_number'] = len(self.all_d)
        self.params['user_number'] = self.data_loader.user_number

        self.model.build_graph_init(self.params)
        self.model.build_graph()
        self.s_prediction = self.model.search_predictions
        self.q_prediction = self.model.question_predictions
        self.train_op = self.model.train_op

    def HIT(self, ground_truth, pred):
        result = []
        print(len(ground_truth))
        print(ground_truth)
        print(pred)
        for k,v in ground_truth.items():
            ground = v
            fit = [i[0] for i in pred[k]][:1]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in ground:
                    tmp += 1
            if tmp > 0:
                result.append(1)
            else:
                result.append(0)
        return np.array(result).mean()

    def MAP(self, ground_truth, pred):
        result = []
        for k,v in ground_truth.items():
            ground = v
            fit = [i[0] for i in pred[k]][:100]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in ground:
                    hit += 1
                    tmp += hit / (j+1)
            result.append(tmp)
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            ground = v
            fit = [i[0] for i in pred[k]][:100]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in ground:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            ground = v
            fit = [i[0] for i in pred[k]][:10]
            temp = 0
            Z_u = 0
            for j in range(len(fit)):
                Z_u = Z_u + 1 / np.log2(j + 2)
                if fit[j] in ground:
                    temp = temp + 1 / np.log2(j + 2)
            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def evaluate(self):
        index_path = self.args.base_path + self.args.category+'output_' + self.args.evaluate + '_result_index'
        index = pd.read_csv(index_path, header=None)
        predictions_path = self.args.base_path + self.args.category+'output_' + self.args.evaluate + '_result'
        predictions = pd.read_csv(predictions_path, header=None)

        ground_truth = {}
        pred = {}

        l = len(predictions.values)
        for i in range(l):
            ind = index.values[i]
            pre = predictions.values[i][0]
            user = ind[0]
            item = ind[1]
            pur_or_not = ind[2]

            if pur_or_not == 1:
                if user not in ground_truth.keys():
                    ground_truth[user] = [item]
                else:
                    ground_truth[user].append(item)

            if user not in pred.keys():
                pred[user] = {item: pre}
            else:
                pred[user][item] = pre

        for k,v in pred.items():
            pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]

        hit = self.HIT(ground_truth, pred)
        map = self.MAP(ground_truth, pred)
        mrr = self.MRR(ground_truth, pred)
        ndcg = self.NDCG(ground_truth, pred)
        return map, mrr, ndcg, hit

    def evaluate_q(self):
        index_path = self.args.base_path + self.args.category+'output_' + self.args.evaluate + '_result_index'
        index = pd.read_csv(index_path, header=None)
        predictions_path = self.args.base_path + self.args.category+'output_' + self.args.evaluate + '_result'
        predictions = pd.read_csv(predictions_path, header=None)

        ground_truth = {}
        pred = {}

        l = len(predictions.values)
        for i in range(l):
            ind = index.values[i]
            pre = predictions.values[i][0]
            user = ind[0]
            item = ind[1]
            ui = str(user)+"@"+str(item)
            ques = ind[2]
            pur_or_not = ind[3]

            if pur_or_not == 1:
                if ui not in ground_truth.keys():
                    ground_truth[ui] = [ques]
                else:
                    ground_truth[ui].append(ques)

            if ui not in pred.keys():
                pred[ui] = {ques: pre}
            else:
                pred[ui][ques] = pre

        for k,v in pred.items():
            pred[k] = sorted(v.items(), key=lambda ques: ques[1])[::-1]

        hit = self.HIT(ground_truth, pred)
        map = self.MAP(ground_truth, pred)
        mrr = self.MRR(ground_truth, pred)
        ndcg = self.NDCG(ground_truth, pred)
        return map, mrr, ndcg, hit

    def run(self):
        with tf.Session() as self.sess:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            best_value = 0.0
            best_result = []
            for epoch in range(self.args.epoch_number):
                for step in range(int(self.train_sample_num/self.args.batch_size)):
                    print('epoch: %s, step %s' % (epoch, step))
                    train_input_fn = self.data_loader.get_train_batch_data(self.args.batch_size)
                    #print(self.all_d[0])
                    #input()
                    self.sess.run(self.train_op, feed_dict={
                        self.model.answer_placeholder: train_input_fn[0],
                        self.model.all_description_placeholder: self.all_d,
                        self.model.pos_description_placeholder: train_input_fn[1],
                        self.model.neg_description_placeholder: train_input_fn[2],
                        self.model.pos_question_placeholder: train_input_fn[3],
                        self.model.neg_question_placeholder: train_input_fn[4],
                        self.model.answer_mask_placeholder: train_input_fn[5],
                        self.model.all_descriptions_mask_placeholder: self.all_d_mask,
                        self.model.pos_descriptions_mask_placeholder: train_input_fn[6],
                        self.model.neg_descriptions_mask_placeholder: train_input_fn[7],
                        self.model.user_placeholder: train_input_fn[8],
                    })

                    if step % 10 == 0:
                        if self.args.evaluate == 'search':
                            result = []
                            for _ in range(self.test_sample_num):
                                test_input_fn = self.data_loader.get_test_batch_data(1)
                                s = self.sess.run(self.s_prediction, feed_dict={
                                    self.model.answer_placeholder: test_input_fn[0],
                                    self.model.pos_description_placeholder: test_input_fn[1],
                                    self.model.pos_descriptions_mask_placeholder: test_input_fn[3],
                                    self.model.user_placeholder: test_input_fn[4]
                                })
                                result += [i.tolist() for i in list(s)]
                            t = pd.DataFrame(result)
                            t.to_csv(self.args.base_path + self.args.category + 'output_'+self.args.evaluate+'_result', index=False,
                                     header=None)
                            map, mrr, ndcg, hit = self.evaluate()
                            if map > best_value:
                                best_result = [map, mrr, ndcg, hit]
                                best_value = map
                            print('map = %s, mrr = %s, ndcg = %s, hit = %s' % (map, mrr, ndcg, hit))
                            print('current best:%s' % (str(best_result)))

                        else:
                            print(str(self.test_sample_num))
                            result = []
                            for _ in range(int(self.test_sample_num/self.args.batch_size)):
                                test_input_fn = self.data_loader.get_test_batch_data(self.args.batch_size)
                                q = self.sess.run(self.q_prediction, feed_dict={
                                    self.model.answer_placeholder: test_input_fn[0],
                                    self.model.all_description_placeholder: self.all_d,
                                    self.model.all_descriptions_mask_placeholder: self.all_d_mask,
                                    self.model.pos_description_placeholder: test_input_fn[1],
                                    self.model.pos_question_placeholder: test_input_fn[2],
                                    self.model.pos_descriptions_mask_placeholder: test_input_fn[3],
                                    self.model.user_placeholder: test_input_fn[4]
                                })
                                result += [i.tolist() for i in list(q)]
                            print('evel')
                            t = pd.DataFrame(result)
                            t.to_csv(self.args.base_path + self.args.category + 'output_'+self.args.evaluate+'_result', index=False,
                                     header=None)
                            map, mrr, ndcg, hit = self.evaluate_q()
                            if map > best_value:
                                best_result = [map, mrr, ndcg, hit]
                                best_value = map
                            print('map = %s, mrr = %s, ndcg = %s, hit = %s' % (map, mrr, ndcg, hit))
                            print('current best:%s' % (str(best_result)))

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("./conf/default_setting.conf")
    parser = argparse.ArgumentParser()
    # 21 parameters
    parser.add_argument('--batch_size', type=int, default=cf.get("parameters", "batch_size"), required=False,
                        help='batch_size')
    parser.add_argument('--use_pretrained', type=int, default=cf.get("parameters", "use_pretrained"), required=False,
                        help='use_pretrained')
    parser.add_argument('--embed_dim', type=int, default=cf.get("parameters", "embed_dim"), required=False,
                        help='embed_dim')
    parser.add_argument('--encoder_type', type=str, default=cf.get("parameters", "encoder_type"), required=False,
                        help='encoder_type')
    parser.add_argument('--cell_type', type=str, default=cf.get("parameters", "cell_type"), required=False,
                        help='cell_type')
    parser.add_argument('--evaluate', type=str, default=cf.get("parameters", "evaluate"), required=False,
                        help='evaluate')
    parser.add_argument('--num_layers', type=int, default=cf.get("parameters", "num_layers"), required=False,
                        help='num_layers')
    parser.add_argument('--num_units', type=int, default=cf.get("parameters", "num_units"), required=False,
                        help='num_units')
    parser.add_argument('--memory_question_hob', type=int, default=cf.get("parameters", "memory_question_hob"), required=False,
                        help='memory_question_hob')
    parser.add_argument('--memory_search_hob', type=int, default=cf.get("parameters", "memory_search_hob"), required=False,
                        help='memory_search_hob')
    parser.add_argument('--dropout', type=float, default=cf.get("parameters", "dropout"), required=False,
                        help='dropout')
    parser.add_argument('--reg_scale', type=float, default=cf.get("parameters", "reg_scale"), required=False,
                        help='reg_scale')
    parser.add_argument('--learning_rate', type=float, default=cf.get("train", "learning_rate"), required=False,
                        help='learning_rate')
    parser.add_argument('--optimizer', type=str, default=cf.get("train", "optimizer"), required=False,
                        help='optimizer')
    parser.add_argument('--train_steps', type=int, default=cf.get("train", "train_steps"), required=False,
                        help='train_steps')
    parser.add_argument('--test_steps', type=int, default=cf.get("train", "test_steps"), required=False,
                        help='test_steps')
    parser.add_argument('--PAD_ID', type=int, default=cf.get("train", "PAD_ID"), required=False,
                        help='PAD_ID')
    parser.add_argument('--model_dir', type=str, default=cf.get("train", "model_dir"), required=False,
                        help='model_dir')
    parser.add_argument('--save_checkpoints_steps', type=int, default=cf.get("train", "save_checkpoints_steps"), required=False,
                        help='save_checkpoints_steps')
    parser.add_argument('--check_hook_n_iter', type=int, default=cf.get("train", "check_hook_n_iter"), required=False,
                        help='check_hook_n_iter')
    parser.add_argument('--min_eval_frequency', type=int, default=cf.get("train", "min_eval_frequency"),
                        required=False, help='min_eval_frequency')
    parser.add_argument('--print_verbose', type=int, default=cf.get("train", "print_verbose"), required=False,
                        help='print_verbose')
    parser.add_argument('--debug', type=int, default=cf.get("train", "debug"), required=False,
                        help='debug')
    parser.add_argument('--category', type=str, default=cf.get("path", "category"), required=False,
                        help='category')
    parser.add_argument('--base_path', type=str, default=cf.get("path", "base_path"), required=False,
                        help='base_path')
    parser.add_argument('--search_with_conversation_number', type=int, default=cf.get("parameters", "search_with_conversation_number"), required=False,
                        help='search_with_conversation_number')
    parser.add_argument('--prediction_with_conversation_number', type=int, default=cf.get("parameters", "prediction_with_conversation_number"), required=False,
                        help='prediction_with_conversation_number')
    parser.add_argument('--epoch_number', type=int, default=cf.get("parameters", "epoch_number"), required=False,
                        help='epoch_number')
    args = parser.parse_args()

    m = Model()
    s = solver(m, args)
    s.run()
