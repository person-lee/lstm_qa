# coding=utf-8

import logging
import datetime
import time
import tensorflow as tf
import operator

from lstm_data_helper import load_train_data, load_test_data, load_embedding, create_valid, batch_iter
from lstm import LSTM


#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../insuranceQA-cnn-lstm/insuranceQA/train", "train corpus file")
tf.flags.DEFINE_string("test_file", "../insuranceQA-cnn-lstm/insuranceQA/test1", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../insuranceQA-cnn-lstm/insuranceQA/vectors.nobin", "embedding file")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding size")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 100, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 300, "epoches")
tf.flags.DEFINE_integer("rnn_size", 300, "embedding size")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "embedding size")
tf.flags.DEFINE_integer("evaluate_every", 3000, "run evaluation")
tf.flags.DEFINE_integer("num_unroll_steps", 100, "embedding size")
tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
tf.flags.DEFINE_float('lr',0.05,'the learning rate')
tf.flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
tf.flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.9, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)
#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
ori_quests, cand_quests = load_train_data(FLAGS.train_file, word2idx, FLAGS.num_unroll_steps)
#train_quests, valid_quests = create_valid(zip(ori_quests, cand_quests))

test_ori_quests, test_cand_quests, labels, results = load_test_data(FLAGS.test_file, word2idx, FLAGS.num_unroll_steps)
#----------------------------------- load data end ----------------------

#----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch, lstm, dropout=1., is_optimizer=True):
    start_time = time.time()
    feed_dict = {
        lstm.ori_input_quests : ori_batch,
        lstm.cand_input_quests : cand_batch, 
        lstm.neg_input_quests : neg_batch,
        lstm.keep_prob : dropout
    }

    if is_optimizer:
        _, step, ori_cand_score, ori_neg_score, cur_loss, cur_acc, ori_seq_len, cand_seq_len, neg_seq_len, ori_cand_dist, ori_neg_dist, ori_quest, cand_quest, neg_quest, out_ori, out_cand, out_neg = sess.run([train_op, global_step, lstm.ori_cand_score, lstm.ori_neg_score, lstm.loss, lstm.acc, lstm.ori_seq_len, lstm.cand_seq_len, lstm.neg_seq_len, lstm.ori_cand_dist, lstm.ori_neg_dist, lstm.ori_quests, lstm.cand_quests, lstm.neg_quests, lstm.out_ori, lstm.out_cand, lstm.out_neg], feed_dict)
    else:
        step, ori_cand_score, ori_neg_score, cur_loss, cur_acc , ori_seq_len, cand_seq_len, neg_seq_len, ori_cand_dist, ori_neg_dist, ori_quest, cand_quest, neg_quest, out_ori, out_cand, out_neg = sess.run([global_step, lstm.ori_cand_score, lstm.ori_neg_score, lstm.loss, lstm.acc, lstm.ori_seq_len, lstm.cand_seq_len, lstm.neg_seq_len, lstm.ori_cand_dist, lstm.ori_neg_dist, lstm.ori_quests, lstm.cand_quests, lstm.neg_quests, lstm.out_ori, lstm.out_cand, lstm.out_neg], feed_dict)


    time_str = datetime.datetime.now().isoformat()
    #logger.info("%s, step %s, loss %s, acc %s"%(time_str, step, cur_loss, cur_acc))
    right, wrong, score = [0.0] * 3
    for i in range(0 ,len(ori_batch)):
        if ori_cand_score[i] > 0.55 and ori_neg_score[i] < 0.4:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    logger.info("%s: step %s, loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch"%(time_str, step, cur_loss, right / (right + wrong), score, wrong, time_elapsed))
    return cur_loss, ori_cand_score
#---------------------------------- execute train model end --------------------------------------

def cal_acc(labels, results, total_ori_cand):
    if len(labels) == len(results) == len(total_ori_cand):
        retdict = {}
        for label, result, ori_cand in zip(labels, results, total_ori_cand):
            if result not in retdict:
                retdict[result] = []
            retdict[result].append((ori_cand, label))
        
        correct = 0
        for key, value in retdict.items():
            value.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = value[0]
            if flag == 1:
                correct += 1
        return 1. * correct/len(retdict)
    else:
        logger.info("data error")
        return 0

#---------------------------------- execute valid model ------------------------------------------
#---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, lstm, valid_ori_quests, valid_cand_quests, labels, results):
    total_loss, idx = 0, 0
    total_ori_cand = []
    #total_right, total_wrong, step = 0, 0, 0, 0
    for ori_valid, cand_valid, neg_valid in batch_iter(valid_ori_quests, valid_cand_quests, FLAGS.batch_size, 1, is_valid=True):
        loss, ori_cand = run_step(sess, ori_valid, cand_valid, cand_valid, lstm, is_optimizer = False)
        total_loss += loss
        total_ori_cand.extend(ori_cand)
        #total_right += right
        #total_wrong += wrong
        idx += 1

    acc = cal_acc(labels, results, total_ori_cand)
    timestr = datetime.datetime.now().isoformat()
    logger.info("%s, evaluation loss:%s, acc:%s"%(timestr, total_loss/idx, acc))
    #logger.info("%s, evaluation loss:%s, acc:%s"%(timestr, total_loss/step, total_right/(total_right + total_wrong)))
#---------------------------------- execute valid model end --------------------------------------
#def inference(lstm, ori_quests, cand_quests, neg_quests):
#    ori_out, cand_out, neg_out = lstm.do_lstm(ori_quests, cand_quests, neg_quests)
#    loss, acc = lstm.cal_loss(ori_out, cand_out, neg_out)
#    return loss, acc

#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:1"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            lstm = LSTM(FLAGS.batch_size, FLAGS.num_unroll_steps, embedding, FLAGS.embedding_size, FLAGS.rnn_size, FLAGS.num_rnn_layers, FLAGS.max_grad_norm)
            global_step = tf.Variable(0, name="globle_step",trainable=False)
            #ori_input_quests = tf.placeholder(tf.int32, shape=[None, FLAGS.num_unroll_steps])
            #cand_input_quests = tf.placeholder(tf.int32, shape=[None, FLAGS.num_unroll_steps])
            #neg_input_quests = tf.placeholder(tf.int32, shape=[None, FLAGS.num_unroll_steps])
            #loss, acc = lstm.do_lstm(ori_input_quests, cand_input_quests, neg_input_quests)
            #train_op = lstm.cal_gradient(lstm.loss, global_step)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                          FLAGS.max_grad_norm)

            optimizer = tf.train.GradientDescentOptimizer(1e-1)
            optimizer.apply_gradients(zip(grads, tvars))
            train_op=optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            sess.run(tf.initialize_all_variables())

            for ori_train, cand_train, neg_train in batch_iter(ori_quests, cand_quests, FLAGS.batch_size, FLAGS.epoches):
                #lr_decay = FLAGS.lr_decay ** max(idx - FLAGS.max_decay_epoch, 0.0)
                #lstm.assign_new_lr(sess, FLAGS.lr*lr_decay)
		run_step(sess, ori_train, cand_train, neg_train, lstm)
                cur_step = tf.train.global_step(sess, global_step)
                
                if cur_step % FLAGS.evaluate_every == 0 and cur_step != 0:
                    logger.info("start to evaluation model")
                    valid_model(sess, lstm, test_ori_quests, test_cand_quests, labels, results)
                    logger.info("evaluation model finish")
            valid_model(sess, lstm, test_ori_quests, test_cand_quests, labels, results)
            #---------------------------------- end train -----------------------------------
