# this will be similar to the train.py file except we will import the live data instead of the historical data 
# so we will need to use web sockets
# keys required
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream

async def trade_callback(t):
    print('trade', t)


async def print_crypto_trade(q):
    print('crypto', q)


# Initiate Class Instance
stream = Stream('PK04UHV69AF2QULV4REU',
                '7g1qUN7qjsfW3U6oSccYYtHyQdoewJ12ANDQmSKd',
                base_url=URL('https://paper-api.alpaca.markets'),
                data_feed='iex')  # <- replace to 'sip' if you have PRO subscription

# subscribing to event
stream.subscribe_crypto_bars(print_crypto_trade, 'BTCUSD')

stream.run()



# somehow load the model and get it to work on the live bar data here