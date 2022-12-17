from keras.models import Sequential,load_model
from keras.layers import Input,Embedding,LSTM,Dropout,BatchNormalization,Dense,Conv1D,Bidirectional,MaxPooling1D,Flatten
import pickle
import pandas as pd
import tweepy
import constants as ct
import re
from textblob import TextBlob
from Tweet import Tweet
import matplotlib.pyplot as plt
import preprocessor as p


m=load_model('SM_CNNLSTM_G.h5')
m.summary()