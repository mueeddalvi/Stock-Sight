import nltk
import re
import sklearn
import tweepy
import numpy as np
import preprocessor as p

from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk import pos_tag,word_tokenize
from sklearn import preprocessing
from Tweet import Tweet

nltk.download('stopwords')
nltk.download('WordLemmatize')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
lemmatizer=WordNetLemmatizer()

def preprocess_text(text):
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
    text=text.encode('ascii', 'ignore').decode('ascii')
    tokens = []
    for token in text.split():
        if  token not in stop_words and len(token)>3:
               tokens.append(lemmatizer.lemmatize(token,get_wordnet_pos(token)))
                
    if len(tokens)<=3:
        return np.nan
    return " ".join(tokens)

def get_wordnet_pos(word):
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={
    'J':wordnet.ADJ,
    'N':wordnet.NOUN,
    'V':wordnet.VERB,
    'R':wordnet.ADV
    }
    return tag_dict.get(tag,wordnet.NOUN)

def getTweets(symbol):
    auth = tweepy.AppAuthHandler(consumer_key='gEnEUVxlBcjAHv3DgnbYnIEmV',
                            consumer_secret='McEFfutfXrcn43qpnd6Hd7g31UR1TqrRBkPRwDWpUItvqTNmH5')


    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search, q='$'+symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(300)

    return tweets


s='''
How Tesla Is Quietly Expanding Its Energy Storage Business https://t.co/E8rfb49Akv via @YouTube $tsla $cciv $nio $xpev $qs $fsr $spce $amzn $aapl $li $gm $f $fcau $enph $sedg $fslr $goog
@LeviathanCapit1 @Post_Market You might want to learn the difference between “worth” and “individual valuation”. Go try selling $AMZN for your “valuation” tomorrow. LMFAO
@LeviathanCapit1 @Post_Market $AMZN is a publicly listed company, so in this instance it means the value the market ascribes to it, today! FFS... using your methodology I’m verging on being a billionaire!
RT @jchatterleyCNN: #Bitcoin: "Everybody gets to participate. It's rules but there's no rulers."
 '''

print(preprocess_text(s)+'\n\n')

print(p.clean(s))

