import pandas_datareader.data as web
import os
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
from datetime import datetime
import pandas_datareader.data as web
import csv
import sys
import math as m

from datetime import datetime
from pandas.tseries.offsets import BDay


class marketGuess(object):
    def __init__(self, company, testSize,
                 n_hidden_states=9, n_latency_days=5,
                 n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10, marketDay = False, days = 0):

        self.marketDay = marketDay
        self.company = company
        self.testSize = testSize
        now = datetime.now()

        dayNumber = now.day
        dayMonth = now.month
        dayYear = now.year

        begin = datetime(dayYear - 4, dayMonth, dayNumber)
        finish = datetime(dayYear, dayMonth, dayNumber)
        finish1 = datetime(dayYear, dayMonth, dayNumber - 1)

        #stock market holiday days
        bdd = ['2019-01-01','2019-01-21','2019-02-18','2019-04-19','2019-05-27','2019-07-04','2019-09-02','2019-11-28','2019-12-25','2018-01-01','2018-01-15','2018-02-19','2018-03-30','2018-05-28','2018-07-04','2018-09-03','2018-12-05','2018-11-22','2018-12-25','2017-01-02','2017-01-16','2017-02-20','2017-04-14','2017-05-29','2017-07-04','2017-10-04','2017-11-23','2017-12-25','2016-01-01',
        '2016-01-18','2016-02-15','2016-03-25','2016-05-30','2016-07-04','2016-09-05','2016-11-24','2016-12-26','2015-01-01','2015-01-19','2015-02-16','2015-04-03','2015-05-25','2015-07-03','2015-09-07','2015-11-26','2015-12-25']
        dateIndex = pd.bdate_range(start = begin, end = finish1, freq = 'C', weekmask = 'Mon Tue Wed Thu Fri', holidays = bdd)

        #create date index with string format
        date = [0] * len(dateIndex)
        for i in range(0, len(dateIndex)):
            date[i] = (dateIndex[i].date()).strftime('%Y-%m-%d')

        #grab API data
        f = web.DataReader(company, 'iex', begin, finish)

        #add approriate dates to dataframe
        f.insert(loc=0, column='date', value=date)

        #Create folder for company data if does not exist
        if not os.path.exists('data/company_data'):
            os.makedirs('data/company_data')


        # write data into csv file
        f.to_csv('data/company_data/{company}.csv'.format(company=company),
        columns=['date', 'open', 'high', 'low', 'close', 'volume'],
        index=False)

        #Checks to see if the market is open today
        #if it is, ask for the open price to attempt to predict the close
        #else let the user knkow
        first = dateIndex[0].date()
        last = dateIndex[(len(dateIndex) - 1)].date()

        if (first <= now.date() <= last):
            self.marketDay = True
            todaysDate = now.date().strftime('%Y-%m-%d')
            todaysOpenPrice = input("\nPlease enter todays opening price for " + str(company) + ': \n')
            todayRow = [todaysDate,todaysOpenPrice, None, None, None, None]
            f.loc[len(f)] = todayRow
            f.to_csv('data/company_data/{company}.csv'.format(company=company),
            columns=['date', 'open', 'high', 'low', 'close', 'volume'],
            index=False)
            self.days = int(round((len(f.index) * testSize))) + 2
        else:
            print("\nThe market is not open today.\n")

        self.days = int(round((len(f.index) * testSize))) + 1

        self.n_latency_days = n_latency_days
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        self.splitLearn(testSize)
        self.probabilities(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def splitLearn(self, testSize):
        data = pd.read_csv(
            'data/company_data/{company}.csv'.format(company=self.company))
        teachData, execData = train_test_split(
            data, test_size=testSize, shuffle=False)

        self.teachData = teachData
        self.execDataSelf = execData

    @staticmethod
    def featureVector(data):
        openPrice = np.array(data['open'])
        closePrice = np.array(data['close'])
        highPrice = np.array(data['high'])
        lowPrice = np.array(data['low'])

        # Compute the fraction change in close, high and low prices
        # which would best be used as a feature
        fracChange = (closePrice - openPrice) / openPrice
        fracHigh = (highPrice - openPrice) / openPrice
        fracLow = (openPrice - lowPrice) / openPrice

        return np.column_stack((fracChange, fracHigh, fracLow))

    def fit(self):
        featureVec = marketGuess.featureVector(self.teachData)
        self.hmm.fit(featureVec)

    def probabilities(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        fracChangeRange = np.linspace(-0.1, 0.1, n_steps_frac_change)
        fracHighRange = np.linspace(0, 0.1, n_steps_frac_high)
        fracLowRange = np.linspace(0, 0.1, n_steps_frac_low)

        self.allResults = np.array(list(itertools.product(fracChangeRange, fracHighRange, fracLowRange)))

    def bestProbability(self, dayIndex):
        prevDataStartIndex = max(0, dayIndex - self.n_latency_days)
        prevDataEndIndex = max(0, dayIndex - 1)
        prevData = self.execDataSelf.iloc[prevDataEndIndex: prevDataStartIndex]
        prevDataFeautures = marketGuess.featureVector(prevData)

        points = []
        for result in self.allResults:
            totData = np.row_stack((prevDataFeautures, result))
            points.append(self.hmm.score(totData))
        bestResult = self.allResults[np.argmax(points)]

        return bestResult

    def predictionClose(self, dayIndex):
        openPrice = self.execDataSelf.iloc[dayIndex]['open']
        predicted_frac_change, _, _ = self.bestProbability(dayIndex)

        return openPrice * (1 + predicted_frac_change)

    def overallPrediction(self, with_plot=False):
        predicted_close_prices = []
        for dayIndex in tqdm(range(self.days)):
            predicted_close_prices.append(self.predictionClose(dayIndex))

        actualClosePrices = []
        for dayIndex in range(self.days):
            actualClosePrices.append(self.execDataSelf.iloc[dayIndex]['close'])

        totGoodPred = 0
        totNotGoodPred = 0
        totDownPred = 0
        totNotDownPred = 0

        goodPred = 0
        notGoodPred = 0
        downPred = 0
        notDownPred = 0

        for i in range(1, len(actualClosePrices)):
            prevClose = actualClosePrices[i - 1]
            prevPred = predicted_close_prices[i - 1]
            actualClose = actualClosePrices[i]
            predClose = predicted_close_prices[i]

            if ((predClose > prevPred) and (actualClose > prevClose)):
                totGoodPred = totGoodPred + 1
            if ((predClose > prevPred) and (actualClose < prevClose)):
                totNotGoodPred = totNotGoodPred + 1
            if ((predClose < prevPred) and (actualClose < prevClose)):
                totDownPred = totDownPred + 1
            if ((predClose < prevPred) and (actualClose > prevClose)):
                totNotDownPred = totNotDownPred + 1
        if (self.days >= 31):
            for i in range((len(actualClosePrices) - 30), len(actualClosePrices)):
                prevClose = actualClosePrices[i - 1]
                prevPred = predicted_close_prices[i - 1]
                actualClose = actualClosePrices[i]
                predClose = predicted_close_prices[i]

                if ((predClose > prevPred) and (actualClose > prevClose)):
                    goodPred = goodPred + 1
                if ((predClose > prevPred) and (actualClose < prevClose)):
                    notGoodPred = notGoodPred + 1
                if ((predClose < prevPred) and (actualClose < prevClose)):
                    downPred = downPred + 1
                if ((predClose < prevPred) and (actualClose > prevClose)):
                    notDownPred = notDownPred + 1

        for i in range(1, len(actualClosePrices)):
            prevClose = actualClosePrices[i - 1]
            prevPred = predicted_close_prices[i - 1]
            actualClose = actualClosePrices[i]
            predClose = predicted_close_prices[i]

            if ((predClose > prevPred) and (actualClose > prevClose)):
                totGoodPred = totGoodPred + 1
            if ((predClose > prevPred) and (actualClose < prevClose)):
                totNotGoodPred = totNotGoodPred + 1
            if ((predClose < prevPred) and (actualClose < prevClose)):
                totDownPred = totDownPred + 1
            if ((predClose < prevPred) and (actualClose > prevClose)):
                totNotDownPred = totNotDownPred + 1
        if (self.days >= 31):
            for i in range((len(actualClosePrices) - 30), len(actualClosePrices)):
                prevClose = actualClosePrices[i - 1]
                prevPred = predicted_close_prices[i - 1]
                actualClose = actualClosePrices[i]
                predClose = predicted_close_prices[i]

                if ((predClose > prevPred) and (actualClose > prevClose)):
                    goodPred = goodPred + 1
                if ((predClose > prevPred) and (actualClose < prevClose)):
                    notGoodPred = notGoodPred + 1
                if ((predClose < prevPred) and (actualClose < prevClose)):
                    downPred = downPred + 1
                if ((predClose < prevPred) and (actualClose > prevClose)):
                    notDownPred = notDownPred + 1

        diff = []
        for i in range(1, len(actualClosePrices)):
            prevClose = actualClosePrices[i - 1]
            prevPred = predicted_close_prices[i - 1]
            actualClose = actualClosePrices[i]
            predClose = predicted_close_prices[i]

            for j in range(self.days):
                diff.append(abs(actualClose - predClose))

        avgDiffSum = 0
        for i in range(0, len(diff)):
            avgDiffSum = avgDiffSum + diff[i]

        avgDiff = round(avgDiffSum/(len(diff) - 1), 2)

        currDiff = []
        if(self.days >= 31):
            for i in range((len(actualClosePrices) - 30), len(actualClosePrices)):
                prevClose = actualClosePrices[i - 1]
                prevPred = predicted_close_prices[i - 1]
                actualClose = actualClosePrices[i]
                predClose = predicted_close_prices[i]


                for j in range(self.days):
                    currDiff.append(abs(actualClose - predClose))

            avgCurrDiffSum = 0
            for i in range(0, len(currDiff)):
                avgCurrDiffSum = avgCurrDiffSum + currDiff[i]

            avgCurrDiff = round(avgCurrDiffSum/(len(currDiff) - 1), 2)

        totIncreaseAccuracy = round((totGoodPred / (totGoodPred + totNotGoodPred))*100, 2)
        totDecreaseAccuracy = round((totDownPred / (totDownPred + totNotDownPred))*100, 2)

        print('\nThe model predicts an increase of ' + str(company) + ' with ' + str(totIncreaseAccuracy) + '% accuracy.')
        print('The model predicts a decrease of ' + str(company) + ' with ' + str(totDecreaseAccuracy) + '% accuracy.')
        print('The average error between the predicted close price and actual close price is ' + '$' + str(avgDiff) + '.')

        if(self.days >= 31):
            increaseAccuracy = round((goodPred / (goodPred + notGoodPred))*100, 2)
            decreaseAccuracy = round((downPred / (downPred + notDownPred))*100, 2)
            print('\nIn the past 30 days the model predicts an increase of ' + str(company) + ' with ' + str(increaseAccuracy) + '% accuracy.')
            print('In the past 30 days the model predicts a decrease of ' + str(company) + ' with ' + str(decreaseAccuracy) + '% accuracy.')

        if(self.days >= 31):
            print('In the past 30 days the average error between the predicted close price and actual close price is $' + str(avgCurrDiff) + '.')

        if(self.marketDay):
            if((predicted_close_prices[len(predicted_close_prices) - 1]) > (predicted_close_prices[len(predicted_close_prices) - 2])):
                print('\nThe model predicts ' + str(company) + ' will increase today.')
            if((predicted_close_prices[len(predicted_close_prices) - 1]) < (predicted_close_prices[len(predicted_close_prices) - 2])):
                print('\nThe model predicts ' + str(company) + ' will decrease today.')

        if with_plot:
            execData = self.execDataSelf[0: self.days]
            self.days = np.array(execData['date'], dtype="datetime64[ms]")
            actual_close_prices = execData['close']

            fig = plt.figure()

            axes = fig.add_subplot(111)
            axes.plot(self.days, actual_close_prices, 'bo-', label="actual")
            axes.plot(self.days, predicted_close_prices, 'r+-', label="predicted")
            axes.set_title('{company}'.format(company=self.company))

            fig.autofmt_xdate()

            plt.legend()
            plt.show()

        return predicted_close_prices

company = input("\nPlease enter a stock symbol (e.g. NFLX): ")

while(True):
    testSize = input("\nFrom the total dataset of 5 years (1007 market days),\nPlease enter the percentage of days you would like to use in decimal form (e.g. 0.05): ")
    testSize = float(testSize)
    if(0.01 <= testSize <= .99):
        break
    else:
        print("\nThat decimal is invalid, please input a decimal between 0.01 and 0.99")
try:
    now = datetime.now()
    dayNumber = now.day
    dayMonth = now.month
    dayYear = now.year
    begin = datetime(dayYear - 4, dayMonth, dayNumber)
    finish = datetime(dayYear, dayMonth, dayNumber)
    f = web.DataReader(company, 'iex', begin, finish)
    print("\nYou will be using " + str(int(round((len(f.index) * testSize))) + 1) + " days.")
except:
    print("Stock symbol is invalid.")

try:
    market_guess = marketGuess(company, testSize)
    market_guess.fit()
    market_guess.overallPrediction(with_plot=True)
except:
    None
