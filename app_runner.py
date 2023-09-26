from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')



import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import pickle
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
nltk.download('punkt')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Input,Embedding,LSTM,Dropout,BatchNormalization,Dense,Conv1D,Bidirectional,MaxPooling1D,Flatten

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
x=[1,2,3,4,5]
app = Flask(__name__)
@app.after_request
def add_header(response):
   response.cache_control.max_age = 0
   return response

@app.after_request
def add_header(response):
   # response.cache_control.no_store = True
   if 'Cache-Control' not in response.headers:
      response.headers['Cache-Control'] = 'no-store'
   return response

@app.route('/')
def index():
   return render_template('indexx.html',title='Home')

@app.route('/predict')
def predict():
   return render_template('predict.html', title='Predict')

@app.route('/tweetsandnews1')
def tweetsandnews1():
   return render_template('tweetsandnews1.html', title='Tweets and News')

@app.route('/knowmore')
def knowmore():
   return render_template('knowmore.html', title='Know More')

@app.route('/livestockstry')
def livestockstry():
   return render_template('livestockstry.html', title='Live Stocks')

@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html', title='About us')

@app.route('/getSymbol', methods=['POST'])
def getSymbol():
   flag=1
   symbol = request.form['search']
   end = datetime.now()
   start = datetime(end.year-5,end.month,end.day)
   data=stock_data(symbol,start,end)
   # today=datetime(end.year,(end.month-6)%,end.day-1)
   df=data.iloc[-1].to_dict()

   # return df
   print(len(df))

   from dateutil.relativedelta import relativedelta
   from datetime import date
   six_months = date.today() + relativedelta(months=-6)
   df1=data.loc[six_months:]

   fig = plt.figure(figsize=(16,8))
   plt.plot(df1['Close'])
   plt.title(symbol)
   plt.savefig(f'static/trend_{symbol}.png')

   return render_template('indexx.html',title='Home', 
                           open=round(df['Open'],4),
                           close=round(df['Close'],4),
                           high=round(df['High'],4),
                           low=round(df['Low'],4),
                           volume=round(df['Volume'],4),
                           flag=flag,symbol=symbol)

@app.route('/predictSymbol', methods=['POST'])
def predictSymbol():
   symbol=request.form['search']
   end = datetime.now()
   start = datetime(end.year-5,end.month,end.day)
   df=stock_data(ticker=symbol,sd=start,ed=end)
   d=df=df[['Close']]
   print(df.head())

   sc=MinMaxScaler(feature_range=(0,1))
   df['Close']=sc.fit_transform(np.array(df['Close']).reshape(-1,1))
   training_size=int(len(df)*.75)
   test_size=int(len(df)-training_size)
   training_data,test_data=df[:training_size] , df[training_size:]
   train=pd.DataFrame(training_data)
   test=pd.DataFrame(test_data)

   x_train,y_train= split_ds(train,7)
   x_test,y_test= split_ds(test,7)
   print(x_train.shape,y_train.shape)
   print(x_test.shape,y_test.shape)

   if(checkModel(symbol) and False):
      print(f'Loading pre trained model for {symbol}')
      m=load_model(f'saved_models/model_{symbol}')

   else:
      print('No model found!!')
      print(f'Training new model for {symbol}...')
      m=model_1(x_train.shape[1:])
      m.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=64,epochs=10)
      # m.save(f'saved_models/model_{symbol}')

   lstm_op=[]
   n=7
   i=0
   x_input=test_data[len(test_data)-7:].to_numpy()
   while(i<7):
      if(len(x_input)>7):
         x_input=x_input[1:]
      pred=m.predict(x_input.reshape(1,-1,1))
      lstm_op.append(pred[0][0])
      x_input=np.append(x_input,pred[0])
      i+=1
   lstm_op=sc.inverse_transform(np.array(lstm_op).reshape(-1,1))
   print(lstm_op)
   lstm_op=np.around(lstm_op,4)
   print(lstm_op)
   print(type(lstm_op))
   print(lstm_op.shape)
   mean=np.mean(lstm_op)

   train_predict=sc.inverse_transform(m.predict(x_train))
   vd=train.loc[train.index[0+x_train.shape[1]]:]
   vd['train_predict']=train_predict
   vd.drop(columns=['Close'],inplace=True)
   print(vd.head())
   df=df.join(vd)

   
   test_predict=sc.inverse_transform(m.predict(x_test))
   vd=test.loc[test.index[0+x_train.shape[1]]:]
   vd['test_predict']=test_predict
   vd.drop(columns=['Close'],inplace=True)
   print(vd.head())
   df=df.join(vd)

   df['Close']=sc.inverse_transform(np.array(df['Close']).reshape(-1,1))

   y_train=sc.inverse_transform(y_train)
   y_test=sc.inverse_transform(y_test)

   err=math.sqrt(mean_squared_error(y_train,train_predict))
   print(f'Error on train {err}')

   err=math.sqrt(mean_squared_error(y_test,test_predict))
   print(f'Error on test {err}')

   print(df.head())
   
   plt.figure(figsize=(16,8))
   plt.title(symbol,fontsize=16)
   plt.plot(df['Close'])
   plt.plot(df['train_predict'])
   plt.plot(df['test_predict'],color='g')
   plt.legend(['Actual','Train Predicted','Test'])
   plt.savefig(f'static/acc_{symbol}.png')
   
   print(err)
   today_stock=df.iloc[-1]['Close']
   global_polarity,tw_list,tw_pol,pos,neg,neutral=retrieving_tweets_polarity1(symbol)
   idea, decision=recommending(df,global_polarity=global_polarity,today_stock=today_stock,mean=mean,quote=symbol)
   return render_template('predict.html',results=lstm_op,
                           flag=1,rmse=round(err,4),decision=decision,symbol=symbol)


@app.route('/getTweets',methods=['POST'])
def getTweets():
   symbol=request.form['search']
   global_polarity,tw_list,tw_pol,pos,neg,neutral=retrieving_tweets_polarity1(symbol)

   return render_template('tweetsandnews1.html',global_polarity=global_polarity,
                           tw_list=tw_list,
                           tw_pol=tw_pol,
                           pos=pos,neg=neg, neutral=neutral,flag=1,symbol=symbol)  

def retrieving_tweets_polarity(symbol):
   stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
   stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
   symbol = stock_full_form['Name'].to_list()[0][0:12]

   auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
   auth.set_access_token(ct.access_token, ct.access_token_secret)
   user = tweepy.API(auth)

   tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)

   tweet_list = [] #List of tweets alongside polarity
   global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
   tw_list=[] #List of tweets only => to be displayed on web page
   #Count Positive, Negative to plot pie chart
   pos=0 #Num of pos tweets
   neg=1 #Num of negative tweets
   for tweet in tweets:
      count=20 #Num of tweets to be displayed on web page
      #Convert to Textblob format for assigning polarity
      tw2 = tweet.full_text
      tw = tweet.full_text
      #Clean
      tw=preprocess_text(tw)
      #print("-------------------------------CLEANED TWEET-----------------------------")
      #print(tw)
      #Replace &amp; by &
      #Remove :
      tw=re.sub(':','',tw)
      #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
      #print(tw)
      #Remove Emojis and Hindi Characters
      tw=tw.encode('ascii', 'ignore').decode('ascii')

      #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
      #print(tw)
      blob = TextBlob(tw)
      polarity = 0 #Polarity of single individual tweet
      for sentence in blob.sentences:

            polarity += sentence.sentiment.polarity
            if polarity>0:
               pos=pos+1
            if polarity<0:
               neg=neg+1

            global_polarity += sentence.sentiment.polarity
      if count > 0:
            tw_list.append(tw2)

      tweet_list.append(Tweet(tw, polarity))
      count=count-1
   if len(tweet_list) != 0:
      global_polarity = global_polarity / len(tweet_list)
   else:
      global_polarity = global_polarity
   neutral=ct.num_of_tweets-pos-neg
   if neutral<0:
      neg=neg+neutral
      neutral=20
   print()
   print("##############################################################################")
   print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
   print("##############################################################################")
   labels=['Positive','Negative','Neutral']
   sizes = [pos,neg,neutral]
   explode = (0, 0, 0)
   fig = plt.figure(figsize=(20,10))
   fig1, ax1 = plt.subplots(figsize=(20,10))
   ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
   # Equal aspect ratio ensures that pie is drawn as a circle
   ax1.axis('equal')
   plt.tight_layout()
   plt.savefig('static/SA.png')
   plt.close(fig)
   #plt.show()
   if global_polarity>0:
      print()
      print("##############################################################################")
      print("Tweets Polarity: Overall Positive")
      print("##############################################################################")
      tw_pol="Overall Positive"
   else:
      print()
      print("##############################################################################")
      print("Tweets Polarity: Overall Negative")
      print("##############################################################################")
      tw_pol="Overall Negative"
   return global_polarity,tw_list,tw_pol,pos,neg,neutral

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def preprocess_text(text):
   from nltk.corpus import stopwords
   stop_words = stopwords.words('english')
   text_remove_symbol='\$[A-Za-z]*'
   text_remove_digits='[0-9]'
   text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
   text_hash=r'#([^\s]+)'
   text=re.sub(text_remove_symbol,'',str(text).lower()).strip()
   text=re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
   text=re.sub(text_hash,'',str(text)).strip()
   text=re.sub(text_remove_digits,'',str(text)).strip()
   text=re.sub('&amp;','&',text)
   text=re.sub(':',' ',text)
   text=text.encode('ascii', 'ignore').decode('ascii')
   tokens = []
   for token in text.split():
      if  token not in stop_words and len(token)>3:
            tokens.append(lemmatizer.lemmatize(token,get_wordnet_pos(token)))
               
   if len(tokens)<=3:
      return np.nan
   return " ".join(tokens)

#RETURNS WHAT PART OF SPEECH IS THE WORD
def get_wordnet_pos(word):
   from nltk.corpus import wordnet
   tag = nltk.pos_tag([word])[0][1][0].upper()
   tag_dict={
   'J':wordnet.ADJ,
   'N':wordnet.NOUN,
   'V':wordnet.VERB,
   'R':wordnet.ADV
   }
   return tag_dict.get(tag, wordnet.NOUN)


def retrieving_tweets_polarity1(symbol):
   # stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
   # stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
   # symbol = stock_full_form['Name'].to_list()[0][0:12]

   auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
   auth.set_access_token(ct.access_token, ct.access_token_secret)
   user = tweepy.API(auth)

   import tensorflow as tf
   from keras.preprocessing.text import Tokenizer
   from keras.preprocessing.sequence import pad_sequences

   pickle_in=open("tokenizer.pkl","rb")
   tokenizer=pickle.load(pickle_in)
   print(type(tokenizer))
   token=tf.keras.preprocessing.text.tokenizer_from_json(tokenizer)
   vocab_size=len(token.word_index.items())+1
   print(vocab_size)

   tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)
   
   print('\n LOADING SENTIMETNT MODEL............ \n')
   m=load_model('SM_CNNLSTM_G.h5')
   m.summary()

   tweet_list = [] #List of tweets alongside polarity
   global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
   tw_list=[] #List of tweets only => to be displayed on web page
   #Count Positive, Negative to plot pie chart
   pos=0 #Num of pos tweets
   neg=1 #Num of negative tweets
   for tweet in tweets:
      count=20 #Num of tweets to be displayed on web page
      #Convert to Textblob format for assigning polarity
      tw2 = tweet.full_text
      tw = tweet.full_text
      #Clean
      tw=preprocess_text(tw)
      # print(type(tw))
      # print(tw)
      polarity = 0 #Polarity of single individual tweet
      # polarity += sentence.sentiment.polarity
      # print(type(tw))
      if(len(str(tw))<7):
         continue
      polarity=m.predict(pad_sequences(token.texts_to_sequences([str(tw)]), maxlen=30))
      if polarity>0.73:
         pos=pos+1
      else:
         neg=neg+1

      global_polarity += polarity

      if count > 0:
         tw_list.append(tw2)

      tweet_list.append(Tweet(tw, polarity))
      count=count-1
   if len(tweet_list) != 0:
      global_polarity = global_polarity / len(tweet_list)
   else:
      global_polarity = global_polarity
   neutral=ct.num_of_tweets-pos-neg
   if neutral<0:
      neg=neg+neutral
      neutral=20
   print()
   print("##############################################################################")
   print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
   print("##############################################################################")


   labels=['Positive','Negative','Neutral']
   sizes = [pos,neg,neutral]
   explode = (0, 0, 0)
   fig = plt.figure()
   fig1, ax1 = plt.subplots()
   ax1.pie(sizes, labels=labels, autopct='%1.2f%%')
   # Equal aspect ratio ensures that pie is drawn as a circle
   ax1.axis('equal')
   plt.tight_layout()
   plt.savefig(f'static/SA_{symbol}.png')
   plt.close(fig)

   #plt.show()
   if global_polarity>0:
      print()
      print("##############################################################################")
      print("Tweets Polarity: Overall Positive")
      print("##############################################################################")
      tw_pol="Public sentiment - Positive"
   else:
      print()
      print("##############################################################################")
      print("Tweets Polarity: Overall Negative")
      print("##############################################################################")
      tw_pol="Public sentiment - Negative"
   return global_polarity,tw_list,tw_pol,pos,neg,neutral


   

def recommending(df, global_polarity,today_stock,mean,quote):
   if today_stock < mean:
      if global_polarity > 0:
            idea="RISE"
            decision="BUY"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
      elif global_polarity <= 0:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
   else:
      idea="FALL"
      decision="SELL"
      print()
      print("##############################################################################")
      print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
   return idea, decision

   

def checkModel(symbol):
   from os import path

   return path.exists(f'saved_models/model_{symbol}')

def stock_data(ticker,sd,ed): #Function which returns stock data for mentioned ticker
   #sd-Start date
   #ed-End date
   #format-yyyy-mm-dd
   data=yf.download(ticker,start=sd,end=ed)
   # data=data[['Close']]
   data.index.name='date'
   d=pd.DataFrame(data=data)
   return d

def pre_model(df):

   from sklearn.preprocessing import MinMaxScaler
   sc=MinMaxScaler(feature_range=(0,1))   #Scaled values btween 0,1
   df.Close=sc.fit_transform(np.array(df['Close']).reshape(-1,1))

   # df.Volume=finData.v_s.fit_transform(np.array(df['Volume']).reshape(-1,1))
   # df.Adj_Close=finData.ac_s.fit_transform(np.array(df['Adj_Close']).reshape(-1,1))

   # df['ma7']=df['Close'].rolling(7).mean()
   df.fillna(0,inplace=True)
   training_size=int(len(df)*.75)
   test_size=int(len(df)-training_size)
   training_data,test_data=df[:training_size] , df[training_size:]
   train=pd.DataFrame(training_data)
   test=pd.DataFrame(test_data)

   return train,test

def split_ds(df,n):
   # n -> window size
   x,y=[],[]
   for i in range (len(df)-n):
      x.append(df.iloc[i:i+n].values)
      y.append(df['Close'][i+n])
   x=np.array(x)
   y=np.array(y)
   y=y.reshape(y.shape[0],1)
   return x,y

def model_1(shape):
   model=Sequential()
   # model.add(Input(shape=(7,3)))
   model.add(LSTM(30, activation='tanh',input_shape=shape, return_sequences=True))
   model.add(LSTM(30,activation='tanh', return_sequences=True))
   model.add(LSTM(30,dropout=0.5, activation='tanh'))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error',optimizer='adam')
   model.summary()
   return model

if __name__ == '__main__':
   app.run(debug=False)
