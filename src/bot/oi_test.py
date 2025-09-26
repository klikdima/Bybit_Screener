import requests
import aiohttp
import asyncio
import logging
import time

# Убери from .bybit_fetch import get_data — если только реально есть этот файл и функция!

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_symbol_cache = {"symbols": [], "timestamp": 0}

def get_all_symbols(min_volume: float = 10_000_000) -> list:
    now = time.time()
    if now - _symbol_cache["timestamp"] < 60:
        return _symbol_cache["symbols"]

    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        logger.warning("❌ Не удалось получить список тикеров Bybit.")
        return _symbol_cache["symbols"]

    symbols_list = data.get("result", {}).get("list", [])
    filtered = []
    for s in symbols_list:
        sym = s.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if any(x in sym for x in ["1000", "USDC", "BUSD", "DYDX", "MEME"]):
            continue
        turnover = s.get("turnover24h") or s.get("turnover_24h", 0)
        try:
            vol = float(turnover)
        except (TypeError, ValueError):
            vol = 0.0
        if vol >= min_volume:
            filtered.append(sym)

    _symbol_cache["symbols"] = filtered
    _symbol_cache["timestamp"] = now
    return filtered

async def get_oi_and_delta(symbol: str, interval: str = "5") -> tuple[float, float]:
    """
    Возвращает (current_open_interest, delta_percent) по Bybit API.
    """
    url = (
        f"https://api.bybit.com/v5/market/open-interest"
        f"?category=linear&symbol={symbol}&interval={interval}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    print(f"\nDEBUG OI for {symbol}: {data}\n")  # Жирно вывожу ответ Bybit!
    lst = data.get("result", {}).get("list", [])
    if len(lst) < 2:
        print("Мало данных в 'list' для расчёта OI. Вероятно, не тот интервал, мало истории или кривой тикер.")
        return 0.0, 0.0
    try:
        oi_prev = float(lst[-2]["openInterest"])
        oi_now  = float(lst[-1]["openInterest"])
        delta_pct = round((oi_now - oi_prev) / oi_prev * 100, 2) if oi_prev else 0.0
        return oi_now, delta_pct
    except Exception as e:
        print(f"Ошибка расчёта OI: {e}")
        return 0.0, 0.0

# ——— Тест запуска прямо из терминала ————
if __name__ == "__main__":
    async def main():
        symbol = "BTCUSDT"
        interval = "5"
        oi, delta = await get_oi_and_delta(symbol, interval)
        print(f"\nOpen Interest: {oi} \nDelta (%): {delta}\n")
    asyncio.run(main())
