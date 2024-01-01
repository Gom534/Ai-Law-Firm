import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec
from numpy import zeros
import tensorflow as tf
from flask import Flask, request, jsonify , send_from_directory
from flask_cors import CORS
from decimal import Decimal
import re
import boto3
from flask import session
from flask import redirect 
from flask_session import Session

app = Flask(__name__)
CORS(app)

# 이부분 keyid수정필요
aws_access_key_id = 'AKIAZU54HSBJGBIDPRWB'
aws_secret_access_key = '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'

# Load the Word2Vec model
model = Word2Vec.load("precetent.model")

# Load the Tokenizer and LSTM model
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = pd.read_pickle('word_index.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
okt = Okt()

# db지역이름 및 테이블 이름 수정필요
region_name = 'ap-northeast-2'  # AWS 리전 이름
table_name = 'Result'  # DynamoDB 테이블 이름
dynamodb = boto3.resource('dynamodb', region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
table = dynamodb.Table(table_name)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract and preprocess the text from the JSON data
  
    사건날짜 = data.get('사건날짜')
    법원 = data.get('법원')
    국선상태 = data.get('국선상태')
    진행방향 = data.get('진행방향')
    교통상황 = data.get('교통상황')
    방향옵션 = data.get('방향옵션')
    선택사항 = data.get('선택사항')
    신호등유무 = data.get('신호등유무')
    사고유무 = data.get('사고유무')
    장애상태 = data.get('장애상태')
    texts = f"{국선상태} {진행방향} {교통상황} {방향옵션} {선택사항} {신호등유무} {사고유무} {장애상태}"
    
    texts = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', texts)
    
    # Tokenize and preprocess the text
    processed_texts = [okt.morphs(text) for text in [texts]]  # Convert single text to a list for consistency

    # Tokenize the text
    sequence = tokenizer.texts_to_sequences(processed_texts)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')

    # Make a prediction using the LSTM model
    probabilities = lstm_model.predict(padded_sequence)

    float_value = float(probabilities[0][0])
    decimal_value = Decimal(float_value)

    item = {
        'id': '1',
        '사건날짜': 사건날짜,
        '법원': 법원,
        '국선상태': 국선상태,
        '진행방향': 진행방향,
        '교통상황': 교통상황,
        '방향옵션': 방향옵션,
        '선택사항': 선택사항,
        '신호등유무': 신호등유무,
        '사고유무': 사고유무,
        '장애상태': 장애상태,
        'probabilities': decimal_value
    }
    table.put_item(Item=item)

    print(decimal_value)
    return jsonify({'result': str(decimal_value)})


@app.route('/getData', methods=['GET'])
def getData():
    response = table.get_item(
    Key={
        'id': '1'
    }
    )
    if 'Item' in response:
        item = response['Item']
        사건날짜 = item.get('사건날짜')
        법원 = item.get('법원')
        국선상태 = item.get('국선상태')
        진행방향 = item.get('진행방향')
        교통상황 = item.get('교통상황')
        방향옵션 = item.get('방향옵션')
        선택사항 = item.get('선택사항')
        신호등유무 = item.get('신호등유무')
        사고유무 = item.get('사고유무')
        장애상태 = item.get('장애상태')
        probabilities = item.get('probabilities')
        # 필요한 작업을 probabilities를 사용하여 수행
        
        print('probabilities:', probabilities)
    else:
        print('해당 항목을 찾을 수 없음')

    return jsonify({
        '사건날짜': 사건날짜,
        '법원': 법원,
        '국선상태': 국선상태,
        '진행방향': 진행방향,
        '교통상황': 교통상황,
        '방향옵션': 방향옵션,
        '선택사항': 선택사항,
        '신호등유무': 신호등유무,
        '사고유무': 사고유무,
        '장애상태': 장애상태,
        'probabilities': str(probabilities)
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



