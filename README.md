# CNN-Stock-Analysis



The purpose of this project is to build and train a Convolutional Neural Network using synthetically generated pictures of stock charts pulled from Alpha Vantage or Google. 

However, taking it a step further we can add together different charts in each quadrant of the synthetically generated pictures of stocks for a more wholeistic image. So instead of just looking at price movements as one chart and then training the CNN on that particular chart, the image being fed into the CNN can be a combination of stock indicators. 

As an example, a synthetically generated image would be split into n quadrants where n is the number of indicator chart patterns we want to use. For this example I want to use the following indicators:

1) Weekly Adjusted Close
2) Simple Moving Average
3) RSI
4) Momentum Indicator 


These indicators would be quilted together to form one picture of 4 quadrants with each quadrant containing the Weekly Adjusted Close Simple Moving Average, RSI, and Momentum Indictor for the CNN to run on. 

For the CNN to work I will need to create thousands of these pictures and categorize them with an outcome I want the CNN to trian itself on. In this case I will use an indicator as to whether the stock price increased (by x%) or decreased (by x%) for the next week. 

By training the CNN on these n indicators I hope to have the CNN weigh in all n features of the dataset instead of simply looking at the price. Choosing which indicators to look at is highly important so I would want a feedback mechanism to see which charts influnce the price the most and weigh them accordingly.
