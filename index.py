from flask import Flask
from flask import render_template
from flask import request
from flask import make_response

from datetime import datetime

import os
import sys
import tensorflow as tf




app = Flask(__name__)





# RNN_Flask.model_load_and_explain();
@app.route("/")
@app.route("/text/rnn")
# @app.route("/rnn", methods=['POST'])
def index():

    x_text_input = request.args.get('text_data')


    static_dir = os.path.abspath(os.path.join(os.curdir, 'static'))
    oi_lime_dir = os.path.abspath(os.path.join(static_dir, 'oi_lime'))
    oi_filename = os.listdir(oi_lime_dir)
    # print("old oi_file name: ", oi_filename)
    if (len(oi_filename) == 0):
        oi_filename = 'No_file'
    else:
        oi_filename = oi_filename[0]
    oi_file_path = os.path.abspath(os.path.join(oi_lime_dir, oi_filename))
    print(oi_file_path)
    if not x_text_input:
        x_text_input = "default"
        # oi_file_path = 'oi_old.html'
        # if(len(oi_filename) == 0):
        #     oi_filename = ''
        # else:
        #     oi_filename = oi_filename[0]
    else:
        tf.reset_default_graph()
        import RNN_Flask
        # RNN_Flask.reset_graph()
        RNN_Flask.restore_model()
        print(x_text_input)
        if os.path.exists(oi_file_path):
            os.remove(oi_file_path)
        oi_filename = RNN_Flask.model_load_and_explain(x_text_input)

    print("final oi_file_path: ",oi_file_path)
    response = make_response(render_template("index.html", oi_filename = oi_filename))
    # response.headers['Last-Modified'] = datetime.now()
    # response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    # response.headers['Pragma'] = 'no-cache'
    # response.headers['Expires'] = '-1'

    return response #render_template("index.html")



@app.route("/text/cnn")
def text_cnn():

    x_text_input = request.args.get('text_data')

    static_dir = os.path.abspath(os.path.join(os.curdir, 'static'))
    oi_lime_dir = os.path.abspath(os.path.join(static_dir, 'oi_lime'))
    oi_filename = os.listdir(oi_lime_dir)
    print("old oi_file name: ", oi_filename)
    if (len(oi_filename) == 0):
        oi_filename = 'No_file'
    else:
        oi_filename = oi_filename[0]
    oi_file_path = os.path.abspath(os.path.join(oi_lime_dir, oi_filename))

    if not x_text_input:
        x_text_input = "default"
        # oi_file_path = 'oi_old.html'
        # if(len(oi_filename) == 0):
        #     oi_filename = ''
        # else:
        #     oi_filename = oi_filename[0]
    else:
        tf.reset_default_graph()
        import CNN_Flask
        # CNN_Flask.reset_graph()
        CNN_Flask.restore_model()
        print(x_text_input)
        if os.path.exists(oi_file_path):
            os.remove(oi_file_path)
        oi_filename = CNN_Flask.model_load_and_explain(x_text_input)

    print("final oi_file_path: ", oi_file_path)
    response = make_response(render_template("text_CNN.html", oi_filename=oi_filename))


    return response  # render_template("index.html")


@app.route("/image")
def index_img():

    # x_text_input = request.args.get('text_data')
    #
    #
    # static_dir = os.path.abspath(os.path.join(os.curdir, 'static'))
    # oi_lime_dir = os.path.abspath(os.path.join(static_dir, 'oi_lime'))
    # oi_filename = os.listdir(oi_lime_dir)
    # print("old oi_file name: ", oi_filename)
    # if (len(oi_filename) == 0):
    #     oi_filename = 'No_file'
    # else:
    #     oi_filename = oi_filename[0]
    # oi_file_path = os.path.abspath(os.path.join(oi_lime_dir, oi_filename))
    #
    # if not x_text_input:
    #     x_text_input = "default"
    #     # oi_file_path = 'oi_old.html'
    #     # if(len(oi_filename) == 0):
    #     #     oi_filename = ''
    #     # else:
    #     #     oi_filename = oi_filename[0]
    # else:
    #     import RNN_Flask
    #     print(x_text_input)
    #     if os.path.exists(oi_file_path):
    #         os.remove(oi_file_path)
    #     oi_filename = RNN_Flask.model_load_and_explain(x_text_input)
    #     # oi_filename = CNN_Flask.model_load_and_explain(x_text_input)
    #
    # print("final oi_file_path: ",oi_file_path)
    response = make_response(render_template("index_img.html"))


    return response #render_template("index.html")



# @app.route("/hello/<name>/<int:age>")
# def hello_name(name,age):
# 	return "Hello, {}, you are {} years old".format(name, age)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port, debug=True)
