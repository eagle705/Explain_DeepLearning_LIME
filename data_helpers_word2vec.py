import nltk
import multiprocessing
from pprint import pprint
import time
import codecs
import re
import os
import numpy as np
import time




def clean_str(string): #데이터 에러때문에 3306번 같은 데이터셋은 아예 날려버려서 안됨
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)


    return string.strip().lower()


def padding_index_of_text(text_data, max_document_length):
    for text in text_data:
        # print(question)
        for i in range(max_document_length-len(text)): #패딩때문에 인덱스 오류나나.. +1 만 더해볼까?
            # print(max_seq_length-len(question))
            text.append(0) # 0번 인덱스가 <PAD>라고 가정함
            # text.append('<PAD>') #어차피 사전에 없어서 패딩 인덱스로 자동 변환됨
        # print(question)
    return text_data

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    pos_file_obj = codecs.open(positive_data_file, "r", 'latin-1')
    positive_examples = list(pos_file_obj.readlines())
    positive_examples = [s.strip() for s in positive_examples]
    neg_file_obj = codecs.open(negative_data_file, "r", 'latin-1')
    negative_examples = list(neg_file_obj.readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    print("num of pos: ",len(positive_labels))
    print("num of neg: ",len(negative_labels))
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_w2v(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    pos_file_obj = codecs.open(positive_data_file, "r", 'latin-1')
    positive_examples = list(pos_file_obj.readlines())
    positive_examples = [s.strip() for s in positive_examples]
    neg_file_obj = codecs.open(negative_data_file, "r", 'latin-1')
    negative_examples = list(neg_file_obj.readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    print(len(positive_labels))
    print(len(negative_labels))
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

# categories = ['alt.atheism', 'soc.religion.christian']
# newsgroups_train = fetch_20newsgroups(subset='train')#, categories=categories)
#class_names = ['atheism', 'christian']
# pprint(newsgroups_train.data)



def make_word2vec():
    print("Loading data...")
    x_text, y = load_data_and_labels_w2v('./rt-polaritydata/rt-polarity.pos', './rt-polaritydata/rt-polarity.neg')
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print(max_document_length)

    doc_list = []
    token_list = []
    token_list.append('<PAD>')
    token_list.append('<START>')
    token_list.append('<EOS>')
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    doc_list.append(token_list)
    from nltk.tokenize import word_tokenize

    for text in x_text:
        doc_list.append([clean_str(str(word)) for word in word_tokenize(text)])



    docs_en = doc_list

    print(len(docs_en))
    print(type(docs_en))
    print(docs_en[5:9])

    # nltk.download('reuters')
    # from nltk.corpus import reuters
    # docs_en2 = [reuters.words(i) for i in reuters.fileids()] #reuters.fileids() == ['test/14826', 'test/14828', ... ,'training/9995']
    #
    # print(len(docs_en2))
    # print(type(docs_en2))
    # print(docs_en2[0:4])


    from gensim.models import word2vec

    config = {
        'min_count': 1,  # 등장 횟수가 5 이하인 단어는 무시
        'size': 100,  # 50차원짜리 벡터스페이스에 embedding
        'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용
        'batch_words': 1000,  # 사전을 구축할때 한번에 읽을 단어 수
        'iter': 30, #7,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수를 의미 #너무 오래 걸릴땐 좀 낮춰야
        'workers': 1 #multiprocessing.cpu_count() #윈도우에서 에러
    }
    wv_model_en = word2vec.Word2Vec(**config)
    wv_model_en.build_vocab(docs_en)

    count_t = time.time()
    wv_model_en.train(docs_en)
    print('Running Time : %.02f' % (time.time() - count_t))

    wv_model_en.save('en_word2vec_MR_100.model')

    try:
        pprint(wv_model_en['i'])
        pprint(wv_model_en.most_similar('i'))
    except:
        pass
    try:
        pprint(wv_model_en['man'])
        pprint(wv_model_en.most_similar('man'))
    except:
        pass


    import json

    vocab_path='./voca.json'
    vocab = dict([(k, v.index) for k, v in wv_model_en.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))


    embedding_weights = wv_model_en.wv.syn0
    final_embeddings = embedding_weights
    labels = wv_model_en.wv.index2word
    print(labels[0])


    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
      assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
      plt.figure(figsize=(18, 18))  # in inches
      for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

      plt.savefig(filename)

    try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt


      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [labels[i] for i in range(plot_only)]
      plot_with_labels(low_dim_embs, labels)




    except ImportError:
      print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector

    embedding_var = tf.Variable(embedding_weights, name='embedding')

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # make log directory if not exists
    log_dir = 'logs/'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    saver.save(sess, save_path=log_dir+'model.ckpt', global_step=None)

    # TODO: add embedding data
    summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # TODO: add label data
    metadata_path = 'logs/metadata.tsv'
    with open(metadata_path, 'w') as f:
        for i, label in enumerate(wv_model_en.wv.index2word):
            f.write('{}\n'.format(label))
    embedding.metadata_path = metadata_path

    # TODO: add sprite image
    # embedding.sprite.image_path = 'mnist/mnist_10k_sprite.png'
    # embedding.sprite.single_image_dim.extend([28, 28])

    # TODO: visualize embedding projector
    projector.visualize_embeddings(summary_writer, config)


# make_word2vec()