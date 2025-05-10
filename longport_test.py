
from time import sleep
from longport.openapi import QuoteContext, Config, SubType, PushQuote

def on_quote(symbol: str, event: PushQuote):
    print(symbol, event)

config = Config.from_env()
ctx = QuoteContext(config)
ctx.set_on_candlestick(on_quote)

while True:
    ctx.subscribe(["700.HK", "AAPL.US"], [
                    SubType.Quote], is_first_push = True)
    sleep(1)

