from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import pprint
from gensim.models import word2vec
import data_helpers_word2vec
tf.set_random_seed(777)  # reproducibility


class TextRNN(object):
    """
     A RNN for text classification.
     """

    def __init__(self, sess, max_document_length, num_classes, embedding_weights, name="RNN"):
        self.sess = sess
        self.max_document_length = max_document_length
        self.num_classes = num_classes
        self.embedding_weights = embedding_weights
        self.net_name = name

        self.current_step = 0

        # self._build_network()

    def last_relevant(self, output, length):  # https://danijar.com/variable-sequence-lengths-in-tensorflow/
        # length는 numpy object~! 일반 list로 하면 안됨
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])

        relevant = tf.gather(flat, index)
        return relevant, index, flat

    def _build_network(self):
        # Placeholder
        self.seq_length_1 = tf.placeholder(tf.int32)
        self.X_train1 = tf.placeholder(tf.int32, [None, max_document_length])  # X data
        self.Y_label = tf.placeholder(tf.float32, [None, num_classes])  # Y label
        self.dropout_keep_prob_lstm = tf.placeholder(tf.float32, name="dropout_keep_prob_lstm")
        self.dropout_keep_prob_fc = tf.placeholder(tf.float32, name="dropout_keep_prob_fc")

        # RNN
        with tf.variable_scope('bi-lstm') as scope:
            lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                                        state_is_tuple=True)
            lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                                        state_is_tuple=True)

            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_fw,
                                                         output_keep_prob=self.dropout_keep_prob_lstm)
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_bw,
                                                         output_keep_prob=self.dropout_keep_prob_lstm)

            # initial_state = lstm_cell.zero_state(batch_size, tf.float32)
            word2vec = tf.nn.embedding_lookup(embedding_weights, self.X_train1)
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,
                                                               lstm_cell_bw,
                                                               word2vec,
                                                               dtype=tf.float32,
                                                               sequence_length=self.seq_length_1)
            output_fw, output_bw = outputs
            states_fw, states_bw = _states

            # tf.get_variable_scope().reuse_variables()

        output_end_fw1, index_fw1, flat_fw1 = self.last_relevant(output_fw, self.seq_length_1)
        output_end_bw1, index_fw1, flat_fw1 = self.last_relevant(output_bw, self.seq_length_1)

        output_concat = tf.concat([output_end_fw1, output_end_bw1], 1)
        # print("output_concat: ", output_concat)

        with tf.name_scope("dropout"):
            output_concat_drop = tf.nn.dropout(output_concat, self.dropout_keep_prob_fc)  # dropout_keep_prob_fc

        with tf.name_scope("FC_layer"):
            X_for_fc = tf.reshape(output_concat_drop, [-1, hidden_size * 2])  # rnn 의 아웃풋 Dim은 어떻게 정하나?
            # fc_w = tf.get_variable(
            #     "fc_w",
            #     shape=[hidden_size*2, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            fc_w = tf.Variable(tf.truncated_normal([hidden_size * 2, num_classes], stddev=0.1), name="fc_w")

            fc_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="fc_b")
            outputs = tf.nn.softmax(tf.matmul(X_for_fc, fc_w) + fc_b, name="predictions")  # 각 time step 당 아웃풋 결과 계산

        # print("Y_label: ", self.Y_label)
        # print("outputs: ", outputs)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Y_label * tf.log(outputs), reduction_indices=[1]))
        self.losses = cross_entropy

        # GradientDescentOptimizer
        self.global_step = tf.Variable(0, trainable=False)  # Opimizer가 업데이트 될때마다 늘어남
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.01)  # .minimize(losses) #RMSPropOptimizer 이게 없다니...
        self._train_op = optimizer.minimize(self.losses, global_step=self.global_step)  # Clipping 안 할때

        # minimize(losses) 를 대체하고 Gradient clipping
        # tvars = tf.trainable_variables()
        # max_grad_norm = 3#5
        # gradients = tf.gradients(losses, tvars)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_grad_norm)
        # _train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=global_step)


        correct_prediction = tf.equal(tf.argmax(self.Y_label, 1), tf.argmax(outputs, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.outputs_prob = outputs




        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)  # tf.train.Saver() 로 선언해도 큰 문제는 없음



    def update(self, x_text_index_batch, label_batch, lengths_batch, output_keep_prob_lstm, out_keep_prob_fc):
        feed_dict = {self.X_train1: x_text_index_batch,
                     self.Y_label: label_batch,
                     self.seq_length_1: lengths_batch,
                     self.dropout_keep_prob_lstm: output_keep_prob_lstm,
                     self.dropout_keep_prob_fc: out_keep_prob_fc}

        self.current_step, acc, loss, _ = self.sess.run(
            [self.global_step, self.accuracy, self.losses, self._train_op], feed_dict=feed_dict)


    def evaluate(self, x_dev, y_dev, x_len_dev, output_keep_prob_lstm=1.0, out_keep_prob_fc=1.0):
        feed_dict_val = {self.X_train1: x_dev,
                         self.Y_label: y_dev,
                         self.seq_length_1: x_len_dev,
                         self.dropout_keep_prob_lstm: output_keep_prob_lstm,
                         self.dropout_keep_prob_fc: out_keep_prob_fc}
        acc, loss = self.sess.run([self.accuracy, self.losses], feed_dict=feed_dict_val)
        print(i, "Evaluation loss:", loss, "Accuracy:", acc)


        outputs_prob_res = self.sess.run(self.outputs_prob, feed_dict=feed_dict_val)

        return outputs_prob_res

    def predict_text_instance(self, x_text):
        x_text_len = [len(x.split(" ")) for x in x_text]  # [len(x_text.split(" "))]
                # print("x_text_len: ", x_text_len)
        x = [(x.split(" ")) for x in x_text]  # [x_text.split(" ")]
        print("x_text_split: ", x)

        x_text_index = word_to_index(x)
        x_text_index_padded = data_helpers_word2vec.padding_index_of_text(x_text_index, max_document_length)
        x_text_index_padded = np.array(x_text_index_padded)
        x_text_len = np.array(x_text_len)


        feed_dict_val = {self.X_train1: x_text_index_padded,
                         self.seq_length_1: x_text_len,
                         self.dropout_keep_prob_lstm: 1.0,
                         self.dropout_keep_prob_fc: 1.0}

        outputs_prob_res = self.sess.run(self.outputs_prob, feed_dict=feed_dict_val)
        # print(i, "Prediction :", outputs_prob_res)
        return outputs_prob_res

    def get_current_step(self):
        return self.current_step

    def save_model(self):
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.current_step)
        print("Saved model checkpoint to {}\n".format(path))

    def restore_model(self):
        self.saver.restore(self.sess, "./model_to_restore/checkpoints/model-1700")
        print("Model restored.")



pp = pprint.PrettyPrinter(indent=4)



# ToDo: Loading data 부분 삭제~!!
# Load data
# print("Loading data...")
# rt_pol_dir = os.path.abspath(os.path.join(os.curdir,'rt-polaritydata'))
#
# pos_path = os.path.abspath(os.path.join(rt_pol_dir, "rt-polarity.pos"))
# neg_path = os.path.abspath(os.path.join(rt_pol_dir, "rt-polarity.neg"))
#
# print(pos_path)
# print(neg_path)
# x_text, y = data_helpers_word2vec.load_data_and_labels(pos_path, neg_path)

# hyper parameters
hidden_size =  300 #3 # RNN hidden size
num_classes = 2  # final output size (RNN or softmax, etc.)
batch_size = 250  # one sample data, one batch
w2v_dim = 100
output_keep_prob_lstm = 0.8
out_keep_prob_fc = 0.8
num_epoch = 40
# total_batch = len(x_textdefault(np.ceil(total_batch / batch_size)) # 올림~ ex) 12/5 = 2.4 -> 3


# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 80

word2vec_pretrained_file = './en_word2vec_MR_100.model'
wv_model_en = word2vec.Word2Vec.load(word2vec_pretrained_file)

word2index = {}
embedding_weights = wv_model_en.wv.syn0
for i,word in enumerate(wv_model_en.wv.index2word): # 다음번엔 0번 자리에 <unk> token 추가해놔야
    word2index[word] = i


def word_to_index(x_text): # 다음부턴 embedding_lookup table을 쓰자.. 패딩도 넣어주고 그게 훨씬낫겟네
    x_text_index = list()
    for sentence in x_text:
        sentence_word_index_list = list()
        for word in sentence:
            if word not in wv_model_en.wv.vocab:
                word_id = 0 # 0번이라는 보장은 없음 <UNK> 토큰의 번호로 해야함
            else:
                word_id = word2index[word]
            sentence_word_index_list.append((word_id))
        x_text_index.append(sentence_word_index_list)
    return x_text_index

# x_text_len = [len(x.split(" ")) for x in x_text]
# x = [(x.split(" ")) for x in x_text] # word단위로 나뉘어짐
#
# x_text_index = word_to_index(x)
# x_text_index_padded = data_helpers_word2vec.padding_index_of_text(x_text_index, max_document_length)
#
# x_text_index_padded = np.array(x_text_index_padded)
# x_text_len = np.array(x_text_len)
# y = np.array(y)


# Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x_text_index_padded[shuffle_indices]
# x_text_len_shuffled = x_text_len[shuffle_indices]
# y_shuffled = y[shuffle_indices]

# Split train/test set
# # TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(0.1 * float(len(y)))
# print("dev_sample_index",dev_sample_index)
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# x_len_train, x_len_dev = x_text_len_shuffled[:dev_sample_index], x_text_len_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#
# x_dev = x_train[0:200]
# x_len_dev = x_len_train[0:200]
# y_dev = y_train[0:200]
#
# print("Vocabulary Size: {:d}".format(len(word2index)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

embedding_weights = np.reshape(embedding_weights,(len(word2index), w2v_dim)) # Voca_size, w2v_Dim


sess = tf.InteractiveSession()

textRNN = TextRNN(sess, max_document_length, num_classes, embedding_weights)
textRNN._build_network()
tf.global_variables_initializer().run()


def restore_model():
    # textRNN._build_network()
    # sess.run(tf.global_variables_initializer())
    textRNN.restore_model()




def reset_graph():
    tf.reset_default_graph()

def model_load_and_explain(x_text_input):

    from lime import lime_text

    print(max_document_length)

    if x_text_input == 'default':
        x_text_instance = '" extreme ops " exceeds expectations . good fun , good action , good acting , good dialogue , good pace , good cinematography .'
    else:
        x_text_instance = x_text_input

    # print(x_text_instance)
    output = textRNN.predict_text_instance([x_text_instance]) #batch -> instnace


    # print (output)

    from lime.lime_text import LimeTextExplainer
    class_names = ['Negative', 'Positive']
    explainer = LimeTextExplainer(class_names=class_names)

    # exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    # print(x_text_instance)
    # print(type(x_text_instance))
    # from sklearn.kernel_ridge import KernelRidge
    # clf = KernelRidge(alpha=1.0)
    # clf.intercept_=''
    # from sklearn.svm import SVR
    # # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # from sklearn.linear_model import Ridge
    # from sklearn.preprocessing import PolynomialFeatures
    # quadratic = PolynomialFeatures(degree=2, interaction_only=True)
    # x_quad_text_instance = quadratic.fit_transform(x_text_instance)


    # from sklearn.linear_model import Ridge
    # from sklearn.preprocessing import PolynomialFeatures
    # quadratic = PolynomialFeatures(degree=2, interaction_only=True)
    # x_text_instance_test = x_text_instance.split(" ")
    # x_text_instance_test_interaction = x_text_instance_test[:]
    # for x_text in x_text_instance_test:
    #     x_text_instance_test_interaction.append(x_text+'')
    #
    # # print("x_text_instance_test1: ",x_text_instance_test)
    # # x_text_instance_test = np.array(x_text_instance_test)
    # # x_text_instance_test = quadratic.fit_transform(x_text_instance_test)
    # print(x_text_instance_test)


    x_text_instance = x_text_instance.strip()

    exp = explainer.explain_instance(x_text_instance, textRNN.predict_text_instance, num_features=6)

    exp.as_list()

    print("")
    print("output prob (Negative, Positive)")
    print('Original prediction:', textRNN.predict_text_instance([x_text_instance])[0])


    print("")
    x_text_removed = x_text_instance
    x_text_removed = x_text_removed.replace(exp.as_list()[0][0], '<unk>')
    x_text_removed = x_text_removed.replace(exp.as_list()[1][0], '<unk>')



    print("x_text_instance: ", x_text_instance)
    print("")
    print("x_text_removed: ", x_text_removed)
    print(exp.as_list()[0][0])
    print(exp.as_list()[1][0])
    print("")



    print('Prediction removing some features:', textRNN.predict_text_instance([x_text_removed])[0])
    print('Difference:', textRNN.predict_text_instance([x_text_instance])[0] - textRNN.predict_text_instance([x_text_removed])[0])

    timestamp = str(int(time.time()))
    static_dir = os.path.abspath(os.path.join(os.curdir, 'static'))
    oi_lime_dir = os.path.abspath(os.path.join(static_dir, 'oi_lime'))
    oi_file_path = os.path.abspath(os.path.join(oi_lime_dir, 'oi_'+timestamp+'.html'))
    exp.save_to_file(oi_file_path)

    return 'oi_'+timestamp+'.html'