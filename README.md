# CNN-Stock-Analysis

The purpose of this project is to build and train a Convolutional Neural Network using synthetically generated pictures of stock charts pulled from Alpha Vantage or Google. 

However, taking it a step further we can add together different charts in each quadrant of the synthetically generated pictures of stocks for a more wholeistic image. So instead of just looking at price movements as one chart (like price movement) and then training the CNN on that particular chart, the image being fed into the CNN can be a combination of stock indicators. This would be analogous to a stock trader looking at price movement and a number of other indicators at the same time before he makes a trade. 

As an example, a synthetically generated image would be split into n quadrants where n is the number of indicator chart patterns we want to use. For this example I want to use the following indicators:

1) Daily Adjusted Close Price: The price that the security ends the day on. 

2) RSI (Relative Strength Index): A momentum indicator used to magnitude of recent price changes to evaluate overbought and oversold conditions. 
https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp

3) AROON Indicator: An indicator designed to determine the trend directon and strength. It has two indicators (AROON up and AROON down) with both numbers ranging from 0 - 100. To combine into one number to use I took the AROON up minus the ARRON down to create an oscillator between -100 and 100. 
https://www.investopedia.com/articles/trading/06/aroon.asp

4) Chaikin A/D line: A volume based indicator designed to measure the cumulative inflow and outflow of money in a security. 0 indicators an inert market, a postive number indicates upward momentum, and a negative number indicates a downward momentum. 
https://www.investopedia.com/articles/active-trading/031914/understanding-chaikin-oscillator.asp

These indicators will be quilted together to form one picture of 4 quadrants with each quadrant containing the Daily Adjusted Close, RSI, AROON Indicator, and the Chaikin A/D line for the CNN to run on. Each picture would take the daily values of price and indicators to form a image representing a full month's worth of data.

For the CNN to work I will need to create thousands of these pictures and categorize them with an outcome I want the CNN to trian itself on. In this case I will use an indicator as to whether the stock price increased (by x%) or decreased (by x%) for the next month. 

By training the CNN on these n indicators I hope to have the CNN weigh in all n features of the dataset instead of simply looking at the price. Choosing which indicators to look at is highly important so I would want a feedback mechanism to see which charts influnce the price the most and weigh them accordingly.
