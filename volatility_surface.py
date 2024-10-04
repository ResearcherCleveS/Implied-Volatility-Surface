import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.title('Implied Volatility Surface for SPY Options')

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol


st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the risk-free rate for the Black-Scholes model.')
risk_free_rate = st.sidebar.number_input('Risk-Free Rate (e.g., 0.015 for 1.5%)', value=0.015)

ticker_symbol = 'SPY'
dividend_yield = 0.013

ticker = yf.Ticker(ticker_symbol)

today = pd.Timestamp('today').normalize()

expirations = ticker.options
exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]

if not exp_dates:
    st.error('No available option expiration dates for SPY.')
else:
    option_data = []

    for exp_date in exp_dates:
        opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
        calls = opt_chain.calls

        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]

        for index, row in calls.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })

    options_df = pd.DataFrame(option_data)

    try:
        spot_history = ticker.history(period='5d')
        if spot_history.empty:
            st.error('Failed to retrieve spot price data for SPY.')
            st.stop()
        else:
            spot_price = spot_history['Close'].iloc[-1]
    except Exception as e:
        st.error(f'An error occurred while fetching spot price data: {e}')
        st.stop()

    options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
    options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

    options_df = options_df[(options_df['strike'] >= spot_price * 0.8) & (options_df['strike'] <= spot_price * 1.2)]

    options_df.reset_index(drop=True, inplace=True)

    options_df['impliedVolatility'] = options_df.apply(
        lambda row: implied_volatility(
            price=row['mid'],
            S=spot_price,
            K=row['strike'],
            T=row['timeToExpiration'],
            r=risk_free_rate,
            q=dividend_yield
        ), axis=1
    )

    options_df.dropna(subset=['impliedVolatility'], inplace=True)

    options_df['impliedVolatility'] *= 100

    options_df.sort_values('strike', inplace=True)

    X = options_df['timeToExpiration'].values  # Time to expiration
    Y = options_df['strike'].values            # Strike price
    Z = options_df['impliedVolatility'].values # Implied volatility

    ti = np.linspace(X.min(), X.max(), 50)
    ki = np.linspace(Y.min(), Y.max(), 50)
    T, K = np.meshgrid(ti, ki)

    Zi = griddata((X, Y), Z, (T, K), method='linear')

    Zi = np.ma.array(Zi, mask=np.isnan(Zi))

    fig = go.Figure(data=[go.Surface(
        x=T, y=K, z=Zi,
        colorscale='Viridis',
        colorbar_title='Implied Volatility (%)'
    )])

    fig.update_layout(
        title='Implied Volatility Surface for SPY Options',
        scene=dict(
            xaxis_title='Time to Expiration (years)',
            yaxis_title='Strike Price ($)',
            zaxis_title='Implied Volatility (%)'
        ),
        autosize=False,
        width=900,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    st.write("---")
    st.markdown(
        "Created by Mateusz Jastrzębski  |   [LinkedIn](https://www.linkedin.com/in/mateusz-jastrz%C4%99bski-8a2622264/) | [GitHub](https://github.com/MateuszJastrzebski21)")

    st.plotly_chart(fig)
