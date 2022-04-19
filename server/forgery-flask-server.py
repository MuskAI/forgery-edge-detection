import numpy as np
from pkg_resources import safe_extra

from time import *
#####api server
import json
from flask import Flask, request, make_response
from flask import jsonify
import urllib.request as urllib
import logging
import cv2
import os
import pdb
import base64
import pdb
app = Flask(__name__)


@app.route('/detect', methods=['GET', 'POST'])
def detecting():
    result_list = []
    try:
        # 读取文件 begin
        file = request.files.get('file')
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, flags=1)
        img = img.astype(np.float32)
        b64_code = base64.b64encode(img)
        resulthtml = {'stage1': b64_code, 'stage2': b64_code}
        response = make_response(jsonify(resulthtml))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'

    except Exception as e:
        print(e)
        logging.error(e)
        result_str = '{"error_id":"505","msg":"请重新提交测试"}'
        response = make_response(jsonify(result_str))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    return response


if __name__ == '__main__':
    app.run(host='192.168.1.137', port=6567, debug=True)
