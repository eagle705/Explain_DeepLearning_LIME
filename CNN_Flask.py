import lime
import sklearn
import sklearn
import sklearn.ensemble
import sklearn.metrics
# from __future__ import print_function
import pprint

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import rnn
import numpy as np
import os
import time
import pprint
from gensim.models import word2vec
import data_helpers_word2vec
tf.set_random_seed(777)  # reproducibility



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sess, sequence_length, num_classes, embedding_weights, embedding_size, filter_sizes, num_filters,
            l2_reg_lambda=0.0):
        self.sess = sess
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_weights = embedding_weights
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda



    def _build_network(self):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = embedding_weights
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # to see softmax
            self.outputs = tf.nn.softmax(self.scores, name="softmax_predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        # Keep track of gradient values and sparsity (optional)


        saver = tf.train.Saver(tf.global_variables())

    def predict_text_instance(self, x_text):
        x_text_len = [len(x.split(" ")) for x in x_text]  # [len(x_text.split(" "))]
        #         print("x_text_len: ", x_text_len)
        x = [(x.split(" ")) for x in x_text]  # [x_text.split(" ")]
        #         print("x: ", x)

        x_text_index = word_to_index(x)
        x_text_index_padded = data_helpers_word2vec.padding_index_of_text(x_text_index, max_document_length)
        x_text_index_padded = np.array(x_text_index_padded)
        x_text_len = np.array(x_text_len)

        #         print("x_text_index_padded: ",x_text_index_padded)
        #         print("x_text_len: ",x_text_len)

        feed_dict_val = {self.input_x: x_text_index_padded,
                         self.dropout_keep_prob: 1.0}

        outputs_prob_res = self.sess.run(self.outputs, feed_dict=feed_dict_val)
        # print(i, "Prediction :", outputs_prob_res)
        return outputs_prob_res

    def get_current_step(self):
        return self.current_step

    def save_model(self):
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.current_step)
        print("Saved model checkpoint to {}\n".format(path))

    def restore_model(self):
        self.saver.restore(self.sess, "./model_to_restore_CNN/checkpoints/model-900")
        print("Model restored.")


pp = pprint.PrettyPrinter(indent=4)


# Load data
# print("Loading data...")
# x_text, y = data_helpers_word2vec.load_data_and_labels('./rt-polaritydata/rt-polarity.pos', './rt-polaritydata/rt-polarity.neg')

# hyper parameters
hidden_size =  300 #3 # RNN hidden size
num_classes = 2  # final output size (RNN or softmax, etc.)
batch_size = 250  # one sample data, one batch
w2v_dim = 100
output_keep_prob_lstm = 0.8
out_keep_prob_fc = 0.8
num_epoch = 40
# total_batch = len(x_text)
# num_iter_per_epoch = int(np.ceil(total_batch / batch_size)) # 올림~ ex) 12/5 = 2.4 -> 3


# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 80

# print(x_text[0:2])
print(max_document_length) # 56


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
# TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(0.1 * float(len(y)))
# print("dev_sample_index",dev_sample_index)
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# x_len_train, x_len_dev = x_text_len_shuffled[:dev_sample_index], x_text_len_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# x_dev = x_train[0:200]
# x_len_dev = x_len_train[0:200]
# y_dev = y_train[0:200]

# print("Vocabulary Size: {:d}".format(len(word2index)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

embedding_weights = np.reshape(embedding_weights,(len(word2index), w2v_dim)) # Voca_size, w2v_Dim



sess = tf.InteractiveSession()


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


import datetime
cnn = TextCNN(
            sess = sess,
            sequence_length=max_document_length,
            num_classes= num_classes,
            embedding_weights=embedding_weights,
            embedding_size=100,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step,  loss, accuracy = sess.run(
        [cnn.train_op, cnn.global_step, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))



def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, loss, accuracy = sess.run(
        [cnn.global_step, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))




cnn._build_network()
sess.run(tf.global_variables_initializer())

def restore_model():

    cnn.restore_model()


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
    output = cnn.predict_text_instance([x_text_instance]) #batch -> instnace



    from lime.lime_text import LimeTextExplainer
    class_names = ['Negative', 'Positive']
    explainer = LimeTextExplainer(class_names=class_names)

    # print(x_text_instance)
    # print(type(x_text_instance))
    exp = explainer.explain_instance(x_text_instance, cnn.predict_text_instance, num_features=6)


    exp.as_list()

    print("")
    print("output prob (Negative, Positive)")
    print('Original prediction:', cnn.predict_text_instance([x_text_instance])[0])


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


    print('Prediction removing some features:', cnn.predict_text_instance([x_text_removed])[0])
    print('Difference:', cnn.predict_text_instance([x_text_instance])[0] - cnn.predict_text_instance([x_text_removed])[0])

    timestamp = str(int(time.time()))
    static_dir = os.path.abspath(os.path.join(os.curdir, 'static'))
    oi_lime_dir = os.path.abspath(os.path.join(static_dir, 'oi_lime'))
    oi_file_path = os.path.abspath(os.path.join(oi_lime_dir, 'oi_'+timestamp+'.html'))
    exp.save_to_file(oi_file_path)

    return 'oi_'+timestamp+'.html'