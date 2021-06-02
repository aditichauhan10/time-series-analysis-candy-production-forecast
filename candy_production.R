#Setting up the required working directory and loading the dataset

setwd("F:/UTA/Sem 4 FALL/ECON/Rcodes")

temp=read.csv("candy-production-data.csv")

#Install and download the required Libraries

#install.packages('ggfortify')
#install.packages('data.table')
#install.packages('TSstudio')
#install.packages('tseries')
library('forecast')
library('ggfortify')
library('data.table')
library('ggplot2')
library('forecast')
library('tseries')
library('urca')
library('TSstudio')
library("xts")

#Creating the required time series and the required holdout time series

candyHOLD=ts(temp[,2],start=c(1972,1),end=c(2020,6),frequency=12)
candy=ts(temp[,2],start=c(1972,1),frequency=12)

candyHOLD

#***************************************Exploratory Data Analysis:

plot(candyHOLD)

# Check for missing values 
sum(is.na(candyHOLD))


# Check the frequency of the time series data 
frequency(candyHOLD)
## [1] 12

#TIME SERIES DECOMPOSITION

decompose_candyts <- decompose(candyHOLD,"multiplicative")
autoplot(decompose_candyts) + theme_classic()


#Let us observe the trend in the data set
trend=ts_plot(candyHOLD, title="Trend in Candy Production")
trend

# Exploring the seasonality of data
seasonality=ts_seasonal(candyHOLD, type="cycle")
seasonality

###############################################################################
# Model 1
#Let us check whether any transformation is required prior to testing:

lambda = BoxCox.lambda(candyHOLD)
lambda
transformData = BoxCox(candyHOLD, lambda)

plot(transformData)

# Since the value of lambda is much greater than 1.
# From the above results, we conclude that any transformation prior to unit
# root testing is not essential

#Let us check if the data is stationary:

# Ho: The unit root is present in the data (The data is not stationary)
# HA: The unit root is not present in the data.(The data is stationary)

summary(ur.df((candyHOLD),lags=49,type="drift",selectlags="AIC"))
summary(ur.df((candyHOLD),lags=48,type="drift",selectlags="Fixed"))

#value of the t-statistic: -1.63 is greater than all the test sizes, we Fail to reject the null hypothesis. 
#we can consider that the unit root is present in the data and non-seasonal differecning is required.

# Let us check whether seasonal differencing is necessary
nsdiffs(candyHOLD)

#For the sake of robustness, we will again check seasonal differencing with 
#first difference of the data
nsdiffs(diff(candyHOLD))

ggtsdisplay(candyHOLD,lag.max=49)

# we can verify it from the graph that there is little or no decay at our seasonal lags.
# Therefore Seasonal differencing is necessary.

# Check for additional differencing:
summary(ur.df(diff(candyHOLD,12),lags=47,type="drift",selectlags="Fixed"))

# According to the above results: t statistic value < 5% test size value.

#Therefore, we do not need more differencing. so, d=1 and D=1

ggtsdisplay((diff(candyHOLD,12)),lag.max=49)

# ACF: Decay with Oscillation
# statistically significant coefficients at 12,24. 
# PACF: statistically significant coefficients at multiples of 12.
# 1st coefficient is also significant.

#SARIMA (p, 1, q) X (P, 1, Dand)
# p: 2,3,4
# q: 1,2
# P: 0,1
# Q: 0,1,2

Arima(candyHOLD,order=c(2,1,1),seasonal=c(0,1,0)) #AIC=3300.19  and   BIC=3317.56
Arima(candyHOLD,order=c(2,1,2),seasonal=c(1,1,2)) #AIC=3145.38  and   BIC=3180.13
Arima(candyHOLD,order=c(2,1,1),seasonal=c(0,1,2)) #AIC=3154.95  and   BIC=3181.02
Arima(candyHOLD,order=c(3,1,2),seasonal=c(0,1,2)) #AIC=3138.31  and   BIC=3173.06
Arima(candyHOLD,order=c(4,1,2),seasonal=c(0,1,2)) #AIC=3139.92  and   BIC=3179.02
Arima(candyHOLD,order=c(4,1,3),seasonal=c(1,1,2)) #AIC=3107.13  and   BIC=3154.91
Arima(candyHOLD,order=c(4,1,3),seasonal=c(0,1,2)) #AIC=3105.05  and   BIC=3148.49

#Selecting the final model with low AIC and BIC values:
modelSARIMA=Arima(candyHOLD,order=c(4,1,3),seasonal=c(0,1,2))
ggtsdisplay(modelSARIMA$residuals,lag.max=49)

checkresiduals(Arima(candyHOLD,order=c(4,1,3),seasonal=c(0,1,2)))

# The Ljung-Box test:

# H0: The first 49 autocorrelations are jointly equal to 0. 
# HA: complement of H0

Box.test(modelSARIMA$residuals,lag=49, type="Ljung-Box")

qchisq(.95,49)

# I fail to reject: 46.892 < 66.338

# 4 step forecast using selected SARIMA model:

foreARIMA=forecast(modelSARIMA,h=4)
foreARIMA

accuracy(foreARIMA,window(candy, start=c(2020,7)))

#Let us create upper limit and lower limit confidence bands
upper=ts(foreARIMA$upper[,2],start=c(2020,7),frequency=12)
lower=ts(foreARIMA$lower[,2],start=c(2020,7),frequency=12)

winDATA=window(candy,start=c(2016,1))

#Plotting the forecast:
plot(cbind(foreARIMA$mean,winDATA,upper,lower),plot.type="single",col=c("BLUE","BLACK", "RED","GREEN"),ylab="FORECAST")

legend("topleft",legend=c("Forecast","Actual Data","Upper","Lower"), col=c("BLUE","BLACK","RED", "GREEN"),lty=c("solid","solid","solid","solid"))

################################################################################################################

# Model 2: Neural Networks 
set.seed(42)
# We performed the Neural Net function on our variable in accordance to the values,
# of p, P and repeats decided by R. That is the default values given by R 

modelNETAR = nnetar(candyHOLD)
foreNN1=forecast(modelNETAR,h=4)
foreNN1

accuracy(foreNN1,window(candy, start=c(2020,7)))

# In the second form of forecast, we decided the values for p,P and repeats,
# and we selected lower values to avoid over fitting of the model.

modelUS = nnetar(candyHOLD,p=2,P=1,size=10,repeats=100)
foreNN2=forecast(modelUS,h=4,PI=TRUE)
foreNN2

accuracy(foreNN2,window(candy, start=c(2020,7)))


upper=ts(foreNN2$upper[,2],start=c(2020,7),frequency=12)
lower=ts(foreNN2$lower[,2],start=c(2020,7),frequency=12)

winDATA=window(candy,start=c(2016,1))

#Plotting the forecast:
plot(cbind(foreNN2$mean,winDATA,upper,lower),plot.type="single",col=c("BLUE","BLACK", "RED","GREEN"),ylab="FORECAST")

legend("topleft",legend=c("Forecast","Actual Data","Upper","Lower"), col=c("BLUE","BLACK","RED", "GREEN"),lty=c("solid","solid","solid","solid"))


# The second model for Neural Net is more accurate in comparison to the first model.

##################################################################################################################
# Among the two models, Neural Net has produced better results than SARIMA.
# Therefore, we'll produce 6 step forecast on our data using Neural Network:

modelUS = nnetar(candy,p=2,P=1,size=10,repeats=100)
finalModel=forecast(modelUS,h=6,PI=TRUE)
finalModel

#Let us create upper limit and lower limit confidence bands
upper=ts(finalModel$upper[,2],start=c(2020,11),frequency=12)
lower=ts(finalModel$lower[,2],start=c(2020,11),frequency=12)

winDATA=window(candy,start=c(2016,1))

#Plotting the forecast:
plot(cbind(finalModel$mean,winDATA,upper,lower),plot.type="single",col=c("BLUE","BLACK", "RED","GREEN"),ylab="FORECAST")

legend("topleft",legend=c("Forecast","Actual Data","Upper","Lower"), col=c("BLUE","BLACK","RED", "GREEN"),lty=c("solid","solid","solid","solid"))
