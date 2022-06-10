from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

# Instância do objeto Flask
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)

# ---------------------------
# Modelo de preços de casa
# ---------------------------
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

# ---------------------------
# Endpoints
# ---------------------------
# Endpoint 1 - entrada
@app.route('/')
def home():
    return 'Minha primeira API.'

# Endpoint 2 - textblob para polaridade de frases (com auth)
@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt-br', to='en')
    polaridade = tb_en.sentiment.polarity
    return f"Polaridade: {polaridade}"

# Endpoint 3 - previsão dos preços de casas
@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco.item())

app.run(debug=True)
