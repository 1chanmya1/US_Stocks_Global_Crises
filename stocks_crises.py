"""
DS2500 Final Class Project
Title: How has major world economic events (Asian Financial Crisis, World
       World Financial Crisis, COVID-19 Pandemic) affected the U.S. Stock
       Market?
Members: Jamie Koo, Theethat Poomijindanon, Chan Nyein
6/22/2023

"""

import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize

class StockData:
    def __init__(self, stocks, start, end):
        self.stocks = stocks
        self.start = start
        self.end = end + pd.offsets.BDay(1)
        self.data = self.download_data()

    def download_data(self):
        """ downloads the adjusted close prices for each stock ticker in the
            list, resamples this data to a monthly frequency, and takes the
            mean of each month """
        return yf.download(self.stocks, self.start, self.end)['Adj Close'].resample('M').last()

    def calculate_returns(self):
        """ calculates the monthly returns for each stock """
        return self.data.pct_change()

    def calculate_cumulative_returns(self):
        """ calculates the cumulative returns for each stock over the entire
            date range """
        return (1 + self.calculate_returns()).cumprod()


class SectorData:
    def __init__(self, sectors, stock_data):
        self.sectors = sectors
        self.stock_data = stock_data
        self.cumulative_returns = self.stock_data.calculate_cumulative_returns()

    def restructure_data(self):
        """ averages the cumulative returns for all stocks in each sector and
            reshapes the data into a long format (each row corresponds to one
            data-sector observation) """
        cumulative_returns_long = pd.DataFrame()
        for sector, stocks in self.sectors.items():
            sector_returns = self.cumulative_returns[stocks].mean(axis=1)
            sector_df = pd.DataFrame(
                {'Date': self.cumulative_returns.index, 'Cumulative Returns': sector_returns, 'Sector': sector})
            cumulative_returns_long = pd.concat([cumulative_returns_long, sector_df], ignore_index=True)
        return cumulative_returns_long

    def plot_data(self):
        """ plots the cumulative returns of each sector over time in a single
            line plot """
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=self.restructure_data(), x='Date', y='Cumulative Returns', hue='Sector')
        plt.grid(True)
        plt.title('Cumulative Returns by Sector')
        plt.show()


class CrisisDataPlotter:
    def __init__(self, crisis_periods, sector_data):
        self.crisis_periods = crisis_periods
        self.sector_data = sector_data
        self.cumulative_returns_long = self.sector_data.restructure_data()

    def plot_crisis_data(self):
        """ plots the cumulative returns of each sector over each crisis
            period in separate subplots (each subplot corresponds to one
            crisis period) """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        for i, (crisis_name, (crisis_start, crisis_end)) in enumerate(self.crisis_periods.items()):
            crisis_returns = self.cumulative_returns_long[(self.cumulative_returns_long['Date'] >= crisis_start) &
                                                          (self.cumulative_returns_long['Date'] <= crisis_end)]
            ax = axes[i]
            sns.lineplot(data=crisis_returns, x='Date', y='Cumulative Returns', hue='Sector', ax=ax)
            ax.grid(True)
            ax.set_title(crisis_name)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig.suptitle('Cumulative Returns across Sectors for each Crisis Period', fontsize=16)
        plt.tight_layout()
        plt.show()


class StockAnalysis:
    def __init__(self, stocks):
        self.stocks = stocks

    def plot_regression(self, stocks_price):
        """ generates a polynomial regression comparing prices of each stock
            with the S&P 500 index """
        fig = plt.figure(figsize=(20, 15), dpi=200)
        rows = 2
        columns = 3
        title = ['AAPL', 'JPM',  'T', 'UNH', 'WMT', 'XOM']
        snp_price = stocks_price['SPY']
        stocks_price = stocks_price.drop(['SPY'], axis=1)

        for i in range(1, len(stocks_price.columns) + 1):
            column = stocks_price.iloc[:, i - 1]
            fig.add_subplot(rows, columns, i)
            plt.grid()

            # polynomial fit with degree = 2
            # code modified from: https://www.statology.org/quadratic-regression-python/
            model = np.poly1d(np.polyfit(snp_price.values.tolist(), column.values.tolist(), 2))
            polyline = np.linspace(1, 500, 50)
            plt.scatter(snp_price.values.tolist(), column.values.tolist())
            plt.plot(polyline, model(polyline))

            #  show equation on plot
            plt.text(300, 1, str(model))

            #  calculate r-squared
            results = {}
            coeffs = np.polyfit(snp_price.values.tolist(), column.values.tolist(), 2)
            p = np.poly1d(coeffs)

            yhat = p(snp_price.values.tolist())
            ybar = np.sum(column.values.tolist()) / len(column.values.tolist())
            ssreg = np.sum((yhat - ybar) ** 2)
            sstot = np.sum((column.values.tolist() - ybar) ** 2)
            results['r_squared'] = ssreg / sstot

            plt.xlabel('SPY')
            plt.ylabel(str(title[i - 1]))
            plt.text(150, 1, 'r2 = 0' + str(round(results['r_squared'], 4)))
            fig.suptitle("SPY vs Stocks in Different Industries", fontsize=28)

        plt.show()
       
class RiskModel:
    def __init__(self, stock_data, period):
        self.stock_data = stock_data.loc[period[0]:period[1]]
        self.period_returns = self.stock_data.pct_change()
        self.period_log_returns = np.log(1 + self.period_returns)

    def calculate_mean(self):
        """ calculates the variance of the log returns """
        return self.period_log_returns.mean()

    def calculate_variance(self):
        """ calculates the variance of the log returns """
        return self.period_log_returns.var()

    def calculate_drift(self):
        """ calculates the 'drift' term which is the expected daily return """
        return self.calculate_mean() - (0.5 * self.calculate_variance())

    def calculate_stddev(self):
        """ calculates the standard deviation of the log returns """
        return self.period_log_returns.std()

    def simulate_future_returns(self, days, simulations):
        """ generates a series of simulated future returns, calculated using
            a stochastic process """
        drift = self.calculate_drift()
        stddev = self.calculate_stddev()
        return (np.exp(drift + stddev * np.random.normal(0, 1, (days, simulations))) - 1) * 100

class CrisisVisualizer:
    def __init__(self, stocks, normal_periods, crisis_periods):
        self.stocks = stocks
        self.normal_periods = normal_periods
        self.crisis_periods = crisis_periods

    def average_returns(self, monthly_normal, monthly_crisis, title, normal_periods, crisis_periods):
        """ generates simulated future returns using risk models for each
            stock, and calculates the mean of these future returns for both
            normal and crisis periods """
        risk_models = {}

        for stock in self.stocks:
            normal_period = normal_periods[title]
            normal_model = RiskModel(monthly_normal[stock], normal_period)
            crisis_period = crisis_periods[title]
            crisis_model = RiskModel(monthly_crisis[stock], crisis_period)
            risk_models[stock] = (normal_model, crisis_model)

        days = 1250
        simulations = 1000
        average_returns_normal = []
        average_returns_crisis = []
        stock_labels = []

        for stock, (model_normal, model_crisis) in risk_models.items():
            future_returns_normal = model_normal.simulate_future_returns(days, simulations)
            average_returns_normal.append(future_returns_normal.mean())

            future_returns_crisis = model_crisis.simulate_future_returns(days, simulations)
            average_returns_crisis.append(future_returns_crisis.mean())

            stock_labels.append(stock)

        return average_returns_normal, average_returns_crisis, stock_labels
   
    def returns_barchart(self, average_returns_normal, average_returns_crisis, stock_labels, title, subplot_axis):
        """ plots a bar chart comparing the average future returns of stocks
            during normal and crisis periods """
        df_normal = pd.DataFrame({'Stock': stock_labels,
                                  'Average Return': average_returns_normal,
                                  'Condition': 'Normal'})

        df_crisis = pd.DataFrame({'Stock': stock_labels,
                                  'Average Return': average_returns_crisis,
                                  'Condition': 'Crisis'})

        df = pd.concat([df_normal, df_crisis])
       
        sns.barplot(data=df, x='Stock', y='Average Return', hue='Condition', ax=subplot_axis)
        subplot_axis.set_title(title)
        subplot_axis.set_ylabel('Average Return (%)')
       
    def calculate_volatility(self, monthly_normal, monthly_crisis, title, normal_periods, crisis_periods):
        """ generates simulated future returns using risk models for each
            stock, and calculates the volatility of these future returns for both
            normal and crisis periods """
        risk_models = {}

        for stock in self.stocks:
            normal_period = normal_periods[title]
            normal_model = RiskModel(monthly_normal[stock], normal_period)
            crisis_period = crisis_periods[title]
            crisis_model = RiskModel(monthly_crisis[stock], crisis_period)
            risk_models[stock] = (normal_model, crisis_model)

        days = 1250
        simulations = 1000
        volatility_normal = []
        volatility_crisis = []
        stock_labels = []

        for stock, (model_normal, model_crisis) in risk_models.items():
            future_returns_normal = model_normal.simulate_future_returns(days, simulations)
            volatility_normal.append(future_returns_normal.std())

            future_returns_crisis = model_crisis.simulate_future_returns(days, simulations)
            volatility_crisis.append(future_returns_crisis.std())

            stock_labels.append(stock)

        return volatility_normal, volatility_crisis, stock_labels
   
    def volatility_barchart(self, volatility_normal, volatility_crisis, stock_labels, title, subplot_axis):
        """ plots a bar chart comparing the volatilities of stocks during
            normal and crisis periods """
        df_normal = pd.DataFrame({'Stock': stock_labels,
                                  'Volatility': volatility_normal,
                                  'Condition': 'Normal'})

        df_crisis = pd.DataFrame({'Stock': stock_labels,
                                  'Volatility': volatility_crisis,
                                  'Condition': 'Crisis'})

        df = pd.concat([df_normal, df_crisis])

        sns.barplot(data=df, x='Stock', y='Volatility', hue='Condition', ax=subplot_axis)
        subplot_axis.set_title(title)
        subplot_axis.set_ylabel('Volatility (%)')
       
class PortfolioOptimization:
    def __init__(self, returns, volatilities=None):
        self.returns = returns
        self.volatilities = volatilities

    def calculate_portfolio_return(self, weights):
        """ calculates monthly return of the portfolio, given a set
            of portfolio weights """
        return np.sum(self.returns.mean() * weights) * 252

    def calculate_portfolio_volatility(self, weights):
        """ calculates the annualized portfolio volatility """
        return np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))

    def calculate_portfolio_risk(self, weights):
        """ calculates total risk of the portfolio by adding the annualized
            portfolio volatility with the volatility of individual stocks """
        if self.volatilities is not None:
            return self.calculate_portfolio_volatility(weights) + np.dot(weights, self.volatilities)
        else:
            return self.calculate_portfolio_volatility(weights)

    def expected_utility(self, weights, risk_aversion):
        """ calculates the expected utility of the portfolio, taking into
            account level of risk aversion """
        return self.calculate_portfolio_return(weights) - 0.5 * risk_aversion * self.calculate_portfolio_risk(weights) ** 2

    def negative_expected_utility(self, weights, risk_aversion):
        """ returns the negative of the expected utility """
        return -self.expected_utility(weights, risk_aversion)

    def optimize_portfolio(self, risk_aversion, initial_weights=None):
        """ finds the portfolio weights that maximize the expected utility of
            the portfolio """
        if risk_aversion < 0 or risk_aversion > 50:
            raise ValueError("Risk aversion must be between 0 and 50.")
       
        num_assets = len(self.returns.columns)
        if initial_weights is None:
            initial_weights = np.repeat(1/num_assets, num_assets)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        boundaries=[(0,1)]
        bounds = tuple(boundaries * len(self.returns.columns))
        optimization_results = minimize(self.negative_expected_utility, initial_weights, args=risk_aversion, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimization_results.x

def main():
    normal_periods = {'Asian Financial Crisis': [dt.datetime(1993, 1, 1), dt.datetime(1996, 12, 31)],
                      'World Financial Crisis': [dt.datetime(2003, 1, 1), dt.datetime(2006, 12, 31)],
                      'COVID-19': [dt.datetime(2016, 1, 1), dt.datetime(2019, 12, 31)]}

    crisis_periods = {'Asian Financial Crisis': (dt.datetime(1997, 7, 2), dt.datetime(1998, 6, 17)),
                      'World Financial Crisis': (dt.datetime(2007, 12, 1), dt.datetime(2009, 6, 1)),
                      'COVID-19': (dt.datetime(2020, 2, 20), dt.datetime(2022, 6, 30))}
   
    # Cumulative Returns by sector
    sectors = {'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
                'Retail': ['WMT', 'HD', 'TGT', 'COST'],
                'Energy': ['XOM', 'CVX', 'BP', 'SHEL'],
                'Healthcare': ['UNH', 'JNJ', 'PFE', 'MRK'],
                'Telecommunications': ['T', 'VZ', 'CMCSA', 'TMUS'],
                'Finance': ['JPM', 'BAC', 'C', 'GS']}
    start = dt.datetime(1993, 1, 1)
    end = dt.datetime(2022, 12, 31)
    lst = [stock for stocks_list in sectors.values() for stock in stocks_list]
    stock_data = StockData(lst, start, end)
    sector_data = SectorData(sectors, stock_data)
    crisis_data_plotter = CrisisDataPlotter(crisis_periods, sector_data)
    crisis_data_plotter.plot_crisis_data()
    sector_data.plot_data()

    # Polynomial Regression
    stocks_spy = ['AAPL', 'WMT', 'XOM', 'UNH', 'T', 'JPM', 'SPY']
    data = StockData(stocks_spy, start, end)
    stocks_prices = data.download_data()
    analysis = StockAnalysis(stocks_prices)
    analysis.plot_regression(stocks_prices)
   
    # Average Future Returns and volatilities simulation
    stocks = ['AAPL', 'WMT', 'XOM', 'UNH', 'T', 'JPM']
    data = StockData(stocks, start, end)
    monthly = data.download_data()
    fig1, axes1 = plt.subplots(1, len(crisis_periods), figsize=(10 * len(crisis_periods), 6))
    fig2, axes2 = plt.subplots(1, len(crisis_periods), figsize=(10 * len(crisis_periods), 6))

    visualizer = CrisisVisualizer(stocks, normal_periods, crisis_periods)

    returns_normal = {}
    returns_cris = {}
    volatility_normal = {}
    volatility_cris = {}
    for i, (crisis, (start_crisis, end_crisis)) in enumerate(crisis_periods.items()):
        monthly_crisis = monthly[start_crisis:end_crisis].resample('M').last()
        normal_period = normal_periods[crisis]
        monthly_normal = monthly[normal_period[0]:normal_period[1]].resample('M').last()
        ret_norm, ret_cris, stock_labels = visualizer.average_returns(monthly_normal, monthly_crisis, crisis, normal_periods,crisis_periods)
        returns_normal[crisis] = ret_norm
        returns_cris[crisis] = ret_cris
        visualizer.returns_barchart(ret_norm, ret_cris, stock_labels, crisis, axes1[i])
        vol_norm, vol_cris, stock_labels = visualizer.calculate_volatility(monthly_normal, monthly_crisis, crisis, normal_periods, crisis_periods)
        volatility_normal[crisis] = vol_norm
        volatility_cris[crisis] = vol_cris
        visualizer.volatility_barchart(vol_norm, vol_cris, stock_labels, crisis, axes2[i])

    fig1.suptitle('Average Future Returns under Normal and Crisis Conditions', fontsize=16)
    fig2.suptitle('Volatility of Future Returns under Normal and Crisis Conditions', fontsize=16)

    plt.tight_layout()
    plt.show()
   
    # Portfolio Optimization
    question1 = input('Are you looking at a Normal period or Crisis period?\n')
    if question1 == 'Normal':
        question2 = int(input('How much risk are you willing to take on on a scale of 0 - 50\n (0 being no risk at all and 50 being a lot of risk)?\n'))
        returns = pd.DataFrame(returns_normal, index = stocks).T
        volatilities_df = pd.DataFrame(volatility_normal, index = stocks).T
        volatilities = volatilities_df.mean().tolist()
        optimizer = PortfolioOptimization(returns, volatilities)
        optimal_weights = optimizer.optimize_portfolio(risk_aversion=question2)
        results = pd.DataFrame(optimal_weights, index = stocks)
        print(results * 100)
    if question1 == 'Crisis':
        question2 = int(input('How much risk are you willing to take on on a scale of 0 - 50\n (0 being no risk at all and 50 being a lot of risk)?\n'))
        returns = pd.DataFrame(returns_cris, index = stocks).T
        volatilities_df = pd.DataFrame(volatility_cris, index = stocks).T
        volatilities = volatilities_df.mean().tolist()
        optimizer = PortfolioOptimization(returns, volatilities)
        optimal_weights = optimizer.optimize_portfolio(risk_aversion=question2)
        results = pd.DataFrame(optimal_weights, index = stocks)
        print(results * 100)
       
main()
