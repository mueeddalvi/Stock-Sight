# Stock Sight
* **Motivation:** Prediction of future movement of stock prices has been a subject matter of many research work. There is a gamut of literature of technical analysis of stock prices where the objective is to identify patterns in stock price movements and derive profit from it but this still remains the most challenging area of research. This system  is a novel method where the predictive module is further augmented by sentiment analysis module that analyses public sentiments in Twitter on the stocks to predict and provide recommendation on purchase. 
* **Objective:**
  * Provide user a one stop experience edifying them about the current and future of the stock movement
  * Provide user a lucid statistics of overall sentiment about an **NASDAQ/NSE** stock communicating a general footing of that stock in market.
  * Allow a novice user to use this as a platform to gain basic information about stock market.
  * Provide user the closing value of NSADAQ/NSE stock over the horizon of a week, i.e. next 7 days.
### Approach:
* Sentiment analysis: A Recurrent Neural Network was trained and later used to predict sentiment by fetching real-time tweets from Twitter
* Stock Forecasting: A LSTM model was trained and later used to predict the **'CLosing Value'** of the stock using retrospective 'Closing Price', 'Volume'
* Outputs from both modules were used to provide an informed recommendation to user on purchase of stocks.
  
### Technologies Used:
* Frontend: HTML, CSS, javascript and jquery
* ML Libraries: Keras, TensorFlow
* APIs: Twitter-api, StockTwits api, Yahoo-Finance-api
* Backend: Flask framework, yfinance library, AlphaVantage timeseries library, sklearn, pandas, tweepy, ggplot library

### Some screenshots:
<br><br>
![image](https://user-images.githubusercontent.com/54925573/208291379-f2a49d1d-cabe-438c-b966-4c7f9be31533.png)
<br><br>
![image](https://user-images.githubusercontent.com/54925573/208291404-088280f5-3ada-45da-b438-a716a0f924fd.png)
<br><br>
![image](https://user-images.githubusercontent.com/54925573/208291413-5b80e3ff-b152-44c2-a0fd-8d26894247d4.png)
<br><br>
![image](https://user-images.githubusercontent.com/54925573/208291419-8a0f1aeb-538a-402c-b158-b33a3d02e4ec.png)
<br><br>


