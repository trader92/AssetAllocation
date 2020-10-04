# -*- coding: utf-8 -*-
import datetime
import time
import json
from datetime import timedelta

import os
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import logging
from pandas_datareader import data
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# user define functions
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{0:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# 필요 parameters 
# 목표하는 자산군 리스트
# 몇년 데이터를 뽑을 지에 대한 데이터
# 돈을 얼마 넣을 것인지
# rebalancing day
# Adj close 사용 여부


useAdjust = False
targets = ['SPY', 'VEA', 'VWO', 'AGG', 'SHY', 'IEF', 'LQD']
# targets = ['SPY', 'EEM', 'EFA', 'AGG', 'SHY', 'IEF', 'LQD']

balance = 10000
initBalance = balance

exceptThisMonth = False

startDateRange = '2001-01-01'
endDateRange = '2020-12-31'
startDateRange = pd.to_datetime(startDateRange)
endDateRange = pd.to_datetime(endDateRange)

# ---------------------------------------

# acc profit
profitPercent = 0.0

# Global var
targetPrices = {}
targetAllPrices = {}

showResult = True
minDateRangeVal = 999999

exceptThisMonth2 = False
if exceptThisMonth:
	exceptThisMonth2 = True

resultSeries = pd.Series()
balanceHistories = pd.DataFrame()

logger.info("- Getting the price...")

# 1. Getting the all prices of last day of month, minimum of the price
for idx, target in enumerate(targets):
	logger.info("Current target : %s" % target)

	d = data.DataReader(target, 'yahoo', '1990-01-01', '2021-12-31')
	print(d)
	prices = d.groupby(pd.Grouper(freq='M'))['Close', 'Adj Close', 'Volume'].tail(1)
	prices2 = d.loc[:, ["Close", "Adj Close", 'Volume']]
	minPrices = d.groupby(pd.Grouper(freq='M'))['Close'].agg(MinClose=np.min)
	minPrices.index = prices.index

	minAdjPrices = d.groupby(pd.Grouper(freq='M'))['Adj Close'].agg(MinAdjClose=np.min)
	minAdjPrices.index = prices.index

	result = pd.concat([prices, minPrices, minAdjPrices], axis=1)
	result = result.sort_values(by='Date', ascending=False)

	# Add the new column for momentum and profit
	for column in [('Profit %s' % (i+1)) for i in range(12)]:
		result[column] = 0.0

	result['Momentum'] = 0.0

	targetPrices[target] = result

	# Add the all data
	targetAllPrices[target] = prices2

logger.info("----- Getting the price complete. -----")
logger.info("----- Calculate the momentum... -----")


# 2. Calculate the momentum
for target in targets:
	logger.info("Current target : %s" % target)

	prices = targetPrices[target]

	# except the 1 year from old data
	for idx, priceData in enumerate(prices.values[:-12]):
		currentPrice = priceData[0]
		prevPrices = [price[0] for price in prices.values[idx+1:idx+13]]

		# profit rate(%)
		profits = [(currentPrice-price)/price * 100 for price in prevPrices]
		momentumVal = (profits[0] * 12) + (profits[2] * 4) + (profits[5] * 2) + profits[-1]

		# save the information
		priceData[4:16] = profits
		priceData[-1] = momentumVal

	# check the minimum date range
	if len(prices) < minDateRangeVal:
		minDateRangeVal = len(prices)

logger.info("----- Calculate the momentum complete. -----")
logger.info("----- Select the position & backtest result -----")
print()

lastEndDay = None
# 3. Select the position
# with minDateRangeVal
for idx in range(minDateRangeVal):
	# ignore first month
	if exceptThisMonth:
		exceptThisMonth = False
		continue

	# Aggressive markets
	aggresiveMarkets = [targetPrices[targets[0]]['Momentum'][idx], targetPrices[targets[1]]['Momentum'][idx], targetPrices[targets[2]]['Momentum'][idx], targetPrices[targets[3]]['Momentum'][idx]]
	defensiveMarkets = [targetPrices[targets[4]]['Momentum'][idx], targetPrices[targets[5]]['Momentum'][idx], targetPrices[targets[6]]['Momentum'][idx]]

	# Check the momentum of aggressive market
	minVal = min(aggresiveMarkets)
	
	# Invest the defensive market
	if minVal < 0:
		maxVal = max(defensiveMarkets)
		maxIdx = defensiveMarkets.index(maxVal) 
		targetMarket = targets[4+maxIdx]
	else: 
		# Invest the aggressive market
		maxVal = max(aggresiveMarkets)
		maxIdx = aggresiveMarkets.index(maxVal) 
		targetMarket = targets[maxIdx]

	if showResult:
		showResult = False
		print("Current target : %s[%.2f]" % (targetMarket, targetPrices[targetMarket]['Close'][idx]))
	else:
		year = str(targetPrices[targetMarket].index[idx-1])[0:4]
		startDay = targetPrices[targetMarket].index[idx]
		endDay = targetPrices[targetMarket].index[idx-1]
		profit = targetPrices[targetMarket]['Profit 1'][idx-1]

		if lastEndDay == None:
			lastEndDay = endDay

		resultSeries[startDay] = targetMarket

		# print("%s to %s result(%s) : %.2f%%" % (startDay, endDay, targetMarket, profit))

# 4. check the profit and maximum drawdown, underwater period
print()
logger.info("----- Protfit results -----")



# remove duplicate result
resultTable = []

for idx in range(len(resultSeries)-1):
	if resultSeries[idx] != resultSeries[idx+1]:
		resultTable.append(idx)

if resultSeries[-1] != resultSeries[-2]:
	resultTable.append(len(resultSeries)-1)

resultSeries = resultSeries[resultTable]

profitMDD = 55555
profitMDDDate = ''
monthMDD = 55555
monthMDDDate = ''


resultSeries2 = pd.Series([], dtype=pd.StringDtype())

# check the profit
for date, market in resultSeries.iteritems():
	if useAdjust:
		buyPrice = targetPrices[market].loc[str(date)]["Adj Close"][0]
		sellPrice = targetPrices[market].loc[str(lastEndDay)]["Adj Close"][0]
		minPrice = targetPrices[market].loc[str(lastEndDay):str(date)]["MinAdjClose"].min()
	else:
		buyPrice = targetPrices[market].loc[str(date)]["Close"][0]
		sellPrice = targetPrices[market].loc[str(lastEndDay)]["Close"][0]
		minPrice = targetPrices[market].loc[str(lastEndDay):str(date)]["MinClose"].min()
	profit = (sellPrice-buyPrice)/buyPrice * 100
	drawdown = (minPrice-buyPrice)/buyPrice * 100

	if profitMDD > profit:
		profitMDD = profit
		profitMDDDate = date

	if monthMDD > drawdown:
		monthMDD = drawdown
		monthMDDDate = date

	# print(date, market, lastEndDay, buyPrice, sellPrice, profit, drawdown)
	print("%s to %s result(%s) : profit[%.2f%%], MMD in month[%.2f%%]" % (date.strftime("%Y-%m-%d"), lastEndDay.strftime("%Y-%m-%d"), market, profit, drawdown))

	# Add the result with date range
	resultSeries2[str(date)] = (market, lastEndDay)

	# change the sell day
	lastEndDay = date


# calculate the balance history
resultSeries2 = resultSeries2.sort_index()

index = 0

for date, data in resultSeries2.iteritems():
	market = data[0]
	endDate = data[1]

	# ignore data
	if not ( pd.to_datetime(date) >= startDateRange and pd.to_datetime(date) <= endDateRange ):
		continue

	if useAdjust:
		prices = targetAllPrices[market].loc[str(date):str(endDate)]["Adj Close"]	
	else:
		prices = targetAllPrices[market].loc[str(date):str(endDate)]["Close"]

	# buy the stock and calc the changes
	count = balance//prices[0]

	# need to add the tax or cost

	remaining =  balance % prices[0]
	balanceChanges = (prices * count) + remaining
	balanceChanges.name = 'balance'
	balanceChangeRate = (balanceChanges/initBalance -1) * 100.0
	balanceChangeRate.name = 'changes'
	balanceChangeRate2 = (balanceChanges/balance -1) * 100.0
	mdd = balanceChangeRate2.min()
	lastProfitRate = balanceChangeRate2[-1]
	
	# last balance
	balance = balanceChanges[-1]

	mddSeries = pd.Series([mdd for i in range(len(balanceChangeRate))], index=balanceChangeRate.index, name="MDD in month")
	profitSeries = pd.Series([lastProfitRate for i in range(len(balanceChangeRate))], index=balanceChangeRate.index, name="profit")
	marketSeries = pd.Series([market for i in range(len(balanceChangeRate))], index=balanceChangeRate.index, name="market")

	# concat the series
	result = pd.concat([balanceChanges, balanceChangeRate, mddSeries, profitSeries, marketSeries], axis=1)
	
	balanceHistories = pd.concat([balanceHistories, result])

firstValue = balanceHistories["balance"][0]
lastValue = balanceHistories["balance"][-1]
year = int((balanceHistories.index[-1] - balanceHistories.index[0]).days/365)

cagr = (pow((lastValue/firstValue), (1/year)) -1) * 100.0

print()
resultMessage = "MDD %.2f%%, monthMDD %.2f%%, CAGR %.2f%%" % (profitMDD, monthMDD, cagr)
print("Profit MDD : %.2f%%[%s], monthMDD : %.2f%%[%s], CAGR : %.2f%%" % (profitMDD, profitMDDDate.strftime("%Y-%m-%d"), monthMDD, monthMDDDate.strftime("%Y-%m-%d"), cagr))

# graph drawing
# graph font
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 16
plt.rcParams["figure.figsize"] = (20, 10)

plt.subplot(2, 2, 1)
plt.plot(balanceHistories.index, balanceHistories['balance'], color='red')
plt.xlabel("기간")
plt.ylabel("잔고")
plt.title("VAA 전략 잔고 변화(%s)" % resultMessage)

# Getting the month profit data
profits = balanceHistories.groupby(pd.Grouper(freq='M'))['profit'].tail(1)
profits = profits.drop_duplicates(keep='last')

plt.subplot(2, 2, 2)
bar = plt.bar(profits.index, profits, color='blue', width=30.0)
# autolabel(bar)
plt.xlabel("기간")
plt.ylabel("각 달의 수익률 변화")


# Getting the month mdd data
mdds = balanceHistories.groupby(pd.Grouper(freq='M'))['MDD in month'].tail(1)

plt.subplot(2, 2, 3)
plt.plot(mdds.index, mdds, color='green')
# autolabel(bar)
plt.xlabel("기간")
plt.ylabel("drawdown in month")

plt.show()

