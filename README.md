# Implied Volatility Surface for SPY Options

A Python application that visualizes the implied volatility surface for SPY options. Using real-time SPY option data from Yahoo Finance, the app calculates implied volatility based on the Black-Scholes model and plots a 3D surface to show how implied volatility varies with time to expiration and strike price. The user-friendly Streamlit interface allows for easy adjustments to model parameters.

## Live Demo on Streamlit
[Access the application on Streamlit](https://volatilitysurface.streamlit.app/)

## Features
- **Interactive 3D Volatility Surface**: View implied volatility changes over varying strike prices and time to expiration.
- **Real-Time Option Data**: The app fetches current SPY options data from Yahoo Finance.
- **Adjustable Risk-Free Rate**: Customize the Black-Scholes model with a user-defined risk-free rate.
- **User-Settable Ticker Symbol**: Input any ticker symbol to visualize its implied volatility surface.
- **Custom Strike Price Filters**: Set minimum and maximum strike price percentages relative to the spot price for tailored analysis.
- **User-Friendly Design**: Built with Streamlit for an accessible and responsive interface.

## Libraries Used
- **Streamlit**: Frontend interface for web-based interaction.
- **yfinance**: Real-time financial data fetching.
- **NumPy**: Numerical computations and array handling.
- **Pandas**: Data management and manipulation.
- **SciPy**: Root-finding and normal distribution functions for calculating implied volatility.
- **Plotly**: 3D plotting and visualization of the implied volatility surface.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software as per the conditions stated in the [MIT License](https://opensource.org/licenses/MIT).

---

Created by [Mateusz Jastrzębski](https://www.linkedin.com/in/mateusz-jastrz%C4%99bski-8a2622264/)
