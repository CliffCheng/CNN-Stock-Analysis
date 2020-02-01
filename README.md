# CNN-Stock-Analysis

Convolutional Neural Networks are most commonly used to recognize animals and animal breeds from pictures.

The purpose of this project is to build and train a Convolutional Neural Network by using synthetically generated pictures of stock charts. 

However, taking it a step further we can add together different charts in each quadrant of the synthetically generated pictures of stocks charts for a more wholeistic image. Instead of looking at price movements as one chart and training a CNN on that particular image, we can train the CNN on a combination of multiple stock indicators. This would be analogous to a stock trader looking at price movement and a number of other indicators at the same time before he makes a trade. 

As an example, a synthetically generated image would be split into n quadrants where n is the number of indicator chart patterns we want to use. For this example I want to use the following indicators:

1) Daily Adjusted Close Price: The price that the security ends the day on. 

2) RSI (Relative Strength Index): A momentum indicator used to magnitude of recent price changes to evaluate overbought and oversold conditions. 
https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp

3) AROON Indicator: An indicator designed to determine the trend directon and strength. It has two indicators (AROON up and AROON down) with both numbers ranging from 0 - 100. To combine into one number to use I took the AROON up minus the ARRON down to create an oscillator between -100 and 100. 
https://www.investopedia.com/articles/trading/06/aroon.asp

4) Chaikin A/D line: A volume based indicator designed to measure the cumulative inflow and outflow of money in a security. 0 indicators an inert market, a postive number indicates upward momentum, and a negative number indicates a downward momentum. 
https://www.investopedia.com/articles/active-trading/031914/understanding-chaikin-oscillator.asp

These indicators will be quilted together to form one picture of 4 quadrants with each quadrant containing the Daily Adjusted Close, RSI, AROON Indicator, and the Chaikin A/D line for the CNN to run on. Each picture would take the daily values of price and indicators to form a single image representing a full month's worth of data.

For the CNN to work I will need to create thousands of these pictures and categorize them with an outcome I want the CNN to trian itself on. In this case I will use an indicator as to whether the stock price increased (by x%) or decreased (by x%) for the next month. 

The project is divided into the following tasks:

1) Obtain a list containing the tickers of companies in the S&P 500 from wikipedia. I will be using BeautifulSoup and requests and saving the list in a pickle file. 

2) Take the ticker list and pull each ticker's worth of data we need from Alpha Vantage (a site that provides free APIs for stocks). Note, Alpha Vantage limits requests to 5 per minute or 500 per day so we will need to create a script to pull the data without triggered a break from Alpha Vantage. 

This is the longest step the process since 500 companies / 1 minute = 8 hour and 33 minutes and may need to be done in batches between days. Therefore I will create another script to keep track of the tickers that are missing and run from there. 

3) Preprocess the data recieved from Alpha Vantage. 

4) Normalize the data to account for varying stock prices.

5) Quilt together the stock indicator images and plot into one figure. 

4) Save the images and categorize them based on whether the next month's data is a gain or a loss. 

5) Train the CNN based off of the images. I will be turn to Google Colab to finish off this part due to needing a much stronger GPU.


By training the CNN on these n indicators I hope to have the CNN weigh in all n features of the dataset instead of simply looking at the price. Choosing which indicators to look at is highly important so I would want a feedback mechanism to see which charts influnce the price the most and weigh them accordingly.
