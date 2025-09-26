import aiohttp
import pandas as pd
from ta.momentum import RSIIndicator

async def get_data(symbol="BTCUSDT", tf="15"):
    try:
        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={tf}&limit=100"
        ticker_url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"

        async with aiohttp.ClientSession() as session:
            # свечи
            async with session.get(kline_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Kline error {resp.status}: {await resp.text()}")
                kl = await resp.json()

            candles = kl.get("result", {}).get("list", [])
            if len(candles) < 21:
                raise ValueError(f"Недостаточно свечей для {symbol}")

            for c in candles:
                if not c[5]:
                    c[5] = "0"

            # тикер
            async with session.get(ticker_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Ticker error {resp.status}: {await resp.text()}")
                tk = await resp.json()

            tl = tk.get("result", {}).get("list", [])
            if not tl:
                raise ValueError(f"Нет тикера для {symbol}")

            last_price = float(tl[0]["lastPrice"])
            fr = tl[0].get("fundingRate")
            funding_rate = float(fr) * 100 if fr is not None else None

        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ]).astype({"open": float, "close": float, "volume": float})
        df.at[df.index[-1], "close"] = last_price

        rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-2]  # по закрытой!
        cur_vol = df["volume"].iloc[-2]        # объем закрытой свечи
        avg_vol = df["volume"].iloc[-22:-2].mean()   # средний за 20 до [-2]
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else None

        op = df["open"].iloc[-2]    # открытие ПРЕДпоследней свечи
        cp = df["close"].iloc[-2]   # закрытие ПРЕДпоследней свечи
        price_change = ((cp - op) / op) * 100 if op > 0 else None

        return {
            "rsi": round(rsi, 2),
            "funding_rate": round(funding_rate, 4) if funding_rate is not None else None,
            "price": round(cp, 4),
            "volume": round(cur_vol, 2),
            "volume_ratio": vol_ratio,
            "price_change": round(price_change, 2) if price_change is not None else None
        }


    except Exception as e:
        print(f"[ERROR] get_data {symbol}: {e}")
        return None

async def get_oi_and_delta(symbol: str, interval: str = "5") -> tuple[float, float]:
    """
    Возвращает текущее Open Interest и %-дельту за последний интервал.
    interval: '5', '15', '30', '60', '240', '720', 'D', 'W', 'M'
    """
    url = (
        "https://api.bybit.com/v5/market/open-interest"
        f"?category=linear&symbol={symbol}&interval={interval}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    print("DEBUG OI:", data)  # <--- Вот это не убирай, чтобы видеть реальные ответы API!

    lst = data.get("result", {}).get("list", [])
    if len(lst) < 2:
        return 0.0, 0.0

    oi_prev = float(lst[-2]["openInterest"])
    oi_now = float(lst[-1]["openInterest"])
    delta_pct = round((oi_now - oi_prev) / oi_prev * 100, 2) if oi_prev else 0.0

    return oi_now, delta_pct
