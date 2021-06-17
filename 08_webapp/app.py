import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import companies

st.write('''
# Turn data into text''')


options = st.multiselect(
     'Please select a stock',
     companies.sp500, help='test')
     # ['Default stock'])

if not options:
   st.stop()

st.write('You selected:', options[0])
st.write("")
st.write("")

stock = options[0]

data = yf.Ticker(stock)

tickerDF = data.history(period = '1d', start = '2019-01-01', end = '2021-01-01')

chart = st.line_chart(tickerDF.Close)

for i in range(len(tickerDF.Open)):
    new_rows = tickerDF.Open[i]
    chart.add_rows(new_rows)
    #progress_bar.progress(i)
    #last_rows = new_rows
    time.sleep(0.05)

my_chart.add_rows(tickerDF.Open)
#st.line_chart(tickerDF.Volume)
with st.beta_expander("See explanation"):
     st.write("""
         Explaning how the piecewise linear representation is done.
     """)


st.write("")
st.write("")



# st.write('You selected:', options)
