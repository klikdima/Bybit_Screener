import requests
import asyncio
import logging
import time
import aiohttp
import re
from collections import Counter
from typing import Optional
from bybit_fetch import get_data
from screener_config import (
    RSI_LOW, RSI_HIGH,
    FUNDING_THRESHOLD, PRICE_THRESHOLD, VOLUME_RATIO_THRESHOLD,
    OI_DELTA_THRESHOLD, IMBALANCE_THRESHOLD, VOLATILITY_THRESHOLD,
    SPOT_FUTURES_SPREAD, MIN_MATCHES
)

logger = logging.getLogger(__name__)
_symbol_cache = {"symbols": [], "timestamp": 0}

def get_tv_deeplink(symbol: str) -> str:
    return f"tradingview://symbol/BYBIT-{symbol}.P"

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
        logger.warning("Не удалось получить список символов с Bybit")
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

async def get_oi_and_delta(symbol: str, interval: str = "5min") -> tuple[float, float]:
    url = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={symbol}&intervalTime={interval}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        lst = data.get("result", {}).get("list", []) if data else []
        if len(lst) < 2:
            return 0.0, 0.0
        oi_prev = float(lst[-2].get("openInterest", 0))
        oi_now = float(lst[-1].get("openInterest", 0))
        delta_pct = round((oi_now - oi_prev) / oi_prev * 100, 2) if oi_prev else 0.0
        return oi_now, delta_pct
    except Exception as e:
        logger.error(f"Ошибка OI {symbol}: {e}")
        return 0.0, 0.0

async def get_spot_derivative_spread(symbol="BTCUSDT"):
    url_deriv = "https://api.bybit.com/v5/market/tickers?category=linear"
    url_spot = "https://api.bybit.com/v5/market/tickers?category=spot"
    price_deriv = price_spot = None

    try:
        async with aiohttp.ClientSession() as session:
            # Деривативы
            async with session.get(url_deriv) as resp:
                d = await resp.json()
                for t in d.get("result", {}).get("list", []):
                    if t.get("symbol") == symbol:
                        price_deriv = float(t.get("lastPrice", 0))
                        break
            # Спот
            async with session.get(url_spot) as resp:
                d = await resp.json()
                for t in d.get("result", {}).get("list", []):
                    if t.get("symbol") == symbol:
                        price_spot = float(t.get("lastPrice", 0))
                        break

        if price_deriv is None or price_spot is None:
            return None, None, None, None

        spread = price_deriv - price_spot
        spread_pct = (spread / price_spot) * 100 if price_spot else 0
        return price_deriv, price_spot, spread, spread_pct
    except Exception as e:
        logger.warning(f"Ошибка spread {symbol}: {e}")
        return None, None, None, None

async def get_orderbook_imbalance(symbol="BTCUSDT", depth=25):
    url = f"https://api.bybit.com/v5/market/orderbook?category=linear&symbol={symbol}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        bids = data.get("result", {}).get("b", [])[:depth]
        asks = data.get("result", {}).get("a", [])[:depth]
        total_bids = sum(float(b[1]) for b in bids)
        total_asks = sum(float(a[1]) for a in asks)
        total = total_bids + total_asks
        if total == 0:
            return None, None
        buy_pct = total_bids / total * 100
        sell_pct = total_asks / total * 100
        return round(buy_pct, 1), round(sell_pct, 1)
    except Exception as e:
        logger.warning(f"Ошибка Imbalance {symbol}: {e}")
        return None, None

async def get_last_closed_candle(symbol: str):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=2"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        # Берём предпоследнюю ([-2]), она уже закрыта!
        candle = data.get("result", {}).get("list", [])[-2]
        return {
            "open": float(candle[1]),
            "close": float(candle[4]),
            "volume": float(candle[5])
        }
    except Exception as e:
        logger.warning(f"Ошибка get_last_closed_candle для {symbol}: {e}")
        return None



async def get_volatility(symbol: str, interval: str = "5", count: int = 20) -> Optional[float]:
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={count}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        candles = data.get("result", {}).get("list", [])
        if len(candles) < count:
            return None
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        ranges = [h - l for h, l in zip(highs, lows)]
        avg_range = sum(ranges) / len(ranges)
        avg_size = sum([(h + l) / 2 for h, l in zip(highs, lows)]) / len(candles)
        if avg_size == 0:
            return None
        volatility = (avg_range / avg_size) * 100
        return volatility
    except Exception as e:
        logger.warning(f"Ошибка вычисления волатильности {symbol}: {e}")
        return None

def escape_markdown(text: str) -> str:
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

# --- Примеры async сканеров ---

from screener_config import PRICE_THRESHOLD

async def scan_price_spikes(price_threshold=PRICE_THRESHOLD):
    symbols = get_all_symbols()
    movers = []
    for symbol in symbols:
        try:
            candle = await asyncio.wait_for(get_last_closed_candle(symbol), timeout=10)
            if not candle:
                continue
            open_price = candle["open"]
            close_price = candle["close"]
            change_pct = (close_price - open_price) / open_price * 100
            if abs(change_pct) >= price_threshold:
                movers.append((symbol, change_pct))
        except Exception as e:
            logger.warning(f"⚠ Ошибка при анализе цены {symbol}: {e}")
    # дальше по стандарту


async def get_rsi_signal(symbol: str):
    timeframes = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    data = {}
    funding = price = volume = volume_ratio = price_change_pct = oi = oi_delta = None

    try:
        d5m = await asyncio.wait_for(get_data(symbol, "5"), timeout=10)
        if not d5m:
            logger.error(f"[{symbol}] get_data(5m) вернул None или пусто")
            for label in timeframes:
                data[label] = None
            fut_spot_str = ""
            imbalance_str = ""
            volatility_str = "н/д"
            price_str = funding_str = volume_str = volume_ratio_str = price_change_str = oi_str = oi_delta_str = "н/д"
        else:
            price = d5m.get("price")
            funding = d5m.get("funding_rate")
            volume = d5m.get("volume")
            volume_ratio = d5m.get("volume_ratio")
            price_change_pct = d5m.get("price_change")
            oi, oi_delta = await get_oi_and_delta(symbol, "5min")
            price_deriv, price_spot, _, _ = await get_spot_derivative_spread(symbol)
            if price_deriv is not None and price_spot is not None:
                deviation = (price_deriv - price_spot) / price_spot * 100 if price_spot else 0
                fut_spot_str = (
                    f"Фьюч: {price_deriv:.2f} | "
                    f"Спот: {price_spot:.2f} | "
                    f"Отклонение: {deviation:+.2f}%\n"
                )
            else:
                fut_spot_str = ""

            buy_pct, sell_pct = await get_orderbook_imbalance(symbol)
            if buy_pct is not None and sell_pct is not None:
                diff_pct = round(buy_pct - sell_pct, 1)
                imbalance_str = (
                    f"Order Book: 🟢{buy_pct:.1f}% Buy / 🔴{sell_pct:.1f}% Sell | "
                    f"Отклонение: {diff_pct:+.1f}%\n"
                )
                if buy_pct > 80:
                    imbalance_str += "⚠️ False push вверх! Осторожно с лонгами.\n"
                elif sell_pct > 80:
                    imbalance_str += "⚠️ False push вниз! Осторожно с шортами.\n"
            else:
                imbalance_str = ""

            volatility = await get_volatility(symbol, interval="5")
            volatility_str = f"{volatility:.2f}%" if volatility is not None else "н/д"

            price_str = f"{price:.4f}" if price is not None else "н/д"
            funding_str = f"{funding:.4f}%" if funding is not None else "н/д ❌"
            volume_str = f"{volume:.2f}" if volume is not None else "н/д"
            volume_ratio_str = f"{volume_ratio:.2f}x" if volume_ratio is not None else "н/д"
            price_change_str = f"{price_change_pct:+.2f}%" if price_change_pct is not None else "н/д"
            oi_str = f"{oi:.0f}" if oi is not None else "н/д"
            oi_delta_str = f"{oi_delta:+.2f}%" if oi_delta is not None else "н/д"
        # Собираем RSI по таймфреймам
        for label, tf in timeframes.items():
            try:
                d = await asyncio.wait_for(get_data(symbol, tf), timeout=10)
                data[label] = d.get("rsi") if d and "rsi" in d else None
            except Exception as e:
                logger.error(f"RSI fetch failed for {symbol} tf={label}: {e}")
                data[label] = None
    except Exception as e:
        logger.error(f"Ошибка сбора данных 5m для {symbol}: {e}")
        for label in timeframes:
            data[label] = None
        fut_spot_str = ""
        imbalance_str = ""
        volatility_str = "н/д"
        price_str = funding_str = volume_str = volume_ratio_str = price_change_str = oi_str = oi_delta_str = "н/д"

    def mark(val):
        if val is None:
            return "нет данных ❌"
        elif val < 30:
            return f"{val:.1f} 🟢 (перепродан)"
        elif val > 70:
            return f"{val:.1f} 🔴 (перекуплен)"
        else:
            return f"{val:.1f} ⚠️"

    token_name = f"{symbol}.P"
    text = (
        f"📊 {symbol} — {price_str} USDT\n"
        f"💸 Funding rate: {funding_str}\n"
        f"{fut_spot_str}"
        f"{imbalance_str}"
        f"📈 Объём: {volume_str} ({volume_ratio_str})\n"
        f"📊 Волатильность (5m): {volatility_str}\n"
        f"📉 Изменение цены (5m): {price_change_str}\n"
        f"📑 OI: {oi_str} ({oi_delta_str})\n"
        f"1m: {mark(data.get('1m'))}\n"
        f"5m: {mark(data.get('5m'))}\n"
        f"15m: {mark(data.get('15m'))}\n"
        f"1h: {mark(data.get('1h'))}\n"
        f"4h: {mark(data.get('4h'))}\n"
        f"1d: {mark(data.get('1d'))}\n"
    )
    print(repr(text))  # оставить как есть
    return text, token_name  # возвращаем два значения!





async def scan_rsi_extremes(return_symbols=False):
    symbols = get_all_symbols()
    oversold = []
    overbought = []
    all_extreme_symbols = []

    for symbol in symbols:
        try:
            d = await asyncio.wait_for(get_data(symbol, "5"), timeout=10)
            if not d or "rsi" not in d:
                continue
            rsi = d["rsi"]
            if rsi < 30:
                oversold.append((symbol, rsi))
                all_extreme_symbols.append(symbol)
            elif rsi > 70:
                overbought.append((symbol, rsi))
                all_extreme_symbols.append(symbol)
        except Exception as e:
            logger.warning(f"⏱ Ошибка RSI сканирования {symbol}: {e}")

    oversold.sort(key=lambda x: x[1])
    overbought.sort(key=lambda x: -x[1])

    result = "📈 RSI сканер (5m)\n"
    result += "\n🟢 Перепроданность (<30):\n"
    result += "\n".join([f"{s}: {r:.1f} 🟢" for s, r in oversold]) if oversold else "- нет -"
    result += "\n\n🔴 Перекупленность (>70):\n"
    result += "\n".join([f"{s}: {r:.1f} 🔴" for s, r in overbought]) if overbought else "- нет -"

    if return_symbols:
        return result, all_extreme_symbols
    return result

from screener_config import FUNDING_THRESHOLD

async def scan_extreme_funding(threshold=FUNDING_THRESHOLD):
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        all_tickers = data.get("result", {}).get("list", [])
    except Exception as e:
        logger.warning(f"⚠ Не удалось получить funding data: {e}")
        return "❌ Ошибка при получении funding rate"

    extreme = []
    for item in all_tickers:
        try:
            symbol = item.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            funding = float(item.get("fundingRate", 0)) * 100
            if abs(funding) >= threshold:
                extreme.append((symbol, funding))
        except Exception:
            continue

    if not extreme:
        return "😴 Нет токенов с аномальным фандингом."

    extreme.sort(key=lambda x: x[1], reverse=True)
    result = "⚡️ Аномальный Funding Rate:\n"
    for s, f in extreme:
        mark = "🔴" if f > 0 else "🟢"
        result += f"{s}: {f:.4f}% {mark}\n"
    return result


from screener_config import PRICE_THRESHOLD

async def scan_price_spikes(price_threshold=PRICE_THRESHOLD):
    symbols = get_all_symbols()
    movers = []
    for symbol in symbols:
        try:
            candle = await asyncio.wait_for(get_last_closed_candle(symbol), timeout=10)
            if not candle:
                continue
            open_price = candle["open"]
            close_price = candle["close"]
            change_pct = (close_price - open_price) / open_price * 100
            if abs(change_pct) >= price_threshold:
                movers.append((symbol, change_pct))
        except Exception as e:
            logger.warning(f"⚠ Ошибка при анализе цены {symbol}: {e}")
    # дальше по стандарту



    if not movers:
        return "😴 Нет токенов с изменением ±2% за 5 минут."

    movers.sort(key=lambda x: abs(x[1]), reverse=True)
    result = "🚨 Движения цены ±2% (5m):\n"
    for s, pc in movers:
        arrow = "🔺" if pc > 0 else "🔻"
        result += f"{s}: {pc:+.2f}% {arrow}\n"
    return result

async def scan_volume_spikes(volume_ratio_threshold=3.0):
    symbols = get_all_symbols()
    spikes = []
    for symbol in symbols:
        try:
            d = await asyncio.wait_for(get_data(symbol, "5"), timeout=10)
            if not d:
                continue
            volume_ratio = d.get("volume_ratio")
            if volume_ratio is None:
                continue
            if volume_ratio >= volume_ratio_threshold:
                spikes.append((symbol, volume_ratio))
        except Exception as e:
            logger.warning(f"⚠ Ошибка анализа объёма {symbol}: {e}")

    if not spikes:
        return "😴 Нет токенов со всплеском объёма."

    spikes.sort(key=lambda x: x[1], reverse=True)
    result = "📊 Всплеск объёма (5m):\n"
    for s, v in spikes:
        result += f"{s}: {v:.2f}x 🔥\n"
    return result

from screener_config import (
    RSI_LOW, RSI_HIGH,
    FUNDING_THRESHOLD, PRICE_THRESHOLD, VOLUME_RATIO_THRESHOLD,
    OI_DELTA_THRESHOLD, IMBALANCE_THRESHOLD, VOLATILITY_THRESHOLD,
    SPOT_FUTURES_SPREAD,
    CUSTOM_MODE, MIN_MATCHES
)

import re

def escape_markdown(text: str) -> str:
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

async def scan_all_indicators(
    rsi_low=RSI_LOW,
    rsi_high=RSI_HIGH,
    funding_threshold=FUNDING_THRESHOLD,
    price_threshold=PRICE_THRESHOLD,
    volume_ratio_threshold=VOLUME_RATIO_THRESHOLD,
    oi_delta_threshold=OI_DELTA_THRESHOLD,
    imbalance_threshold=IMBALANCE_THRESHOLD,
    volatility_threshold=VOLATILITY_THRESHOLD,
    spot_futures_spread=SPOT_FUTURES_SPREAD,
    min_matches=MIN_MATCHES
):
    symbols = get_all_symbols()
    results = {
        "ANOMALY_LONG": [],
        "ANOMALY_SHORT": [],
        "VOLUME_PLUS_ANY": [],
        "MIN_MATCHES": []
    }

    async def analyze_symbol(symbol):
        alerts = []
        score = 0
        volume_spike = False
        other_signals = 0

        try:
            d = await asyncio.wait_for(get_data(symbol, "5"), timeout=10)
            if not d:
                return None
        except Exception as e:
            logger.warning(f"Ошибка get_data {symbol}: {e}")
            return None

        rsi = d.get("rsi")
        funding = d.get("funding_rate")
        price_change = d.get("price_change")
        volume_ratio = d.get("volume_ratio")

        if volume_ratio is not None and volume_ratio >= volume_ratio_threshold:
            alerts.append(f"Volume x{volume_ratio:.2f} 🔥")
            volume_spike = True
            score += 1

        if rsi is not None:
            if rsi < rsi_low:
                alerts.append(f"RSI {rsi:.1f} 🟢")
                other_signals += 1
                score += 1
            elif rsi > rsi_high:
                alerts.append(f"RSI {rsi:.1f} 🔴")
                other_signals += 1
                score += 1

        if funding is not None and abs(funding) >= funding_threshold:
            mark = "🔴" if funding > 0 else "🟢"
            alerts.append(f"Funding {funding*100:+.2f}% {mark}")
            other_signals += 1
            score += 1

        if price_change is not None and abs(price_change) >= price_threshold:
            arrow = "🔺" if price_change > 0 else "🔻"
            alerts.append(f"Price {price_change:+.2f}% {arrow}")
            other_signals += 1
            score += 1

        oi, oi_delta = await get_oi_and_delta(symbol, "5min")
        if abs(oi_delta) >= oi_delta_threshold:
            arrow = "↑" if oi_delta > 0 else "↓"
            alerts.append(f"OI Δ {oi_delta:+.2f}% {arrow}")
            other_signals += 1
            score += 1

        buy_pct, sell_pct = await get_orderbook_imbalance(symbol)
        if buy_pct is not None and (buy_pct >= imbalance_threshold or sell_pct >= imbalance_threshold):
            side = "Buy" if buy_pct > sell_pct else "Sell"
            alerts.append(f"Imbalance {buy_pct:.1f}/{sell_pct:.1f} ({side})")
            other_signals += 1
            score += 1

        volatility = await get_volatility(symbol, interval="5")
        if volatility is not None and volatility >= volatility_threshold:
            alerts.append(f"Volatility {volatility:.2f}%")
            other_signals += 1
            score += 1

        price_deriv, price_spot, spread, spread_pct = await get_spot_derivative_spread(symbol)
        if spread_pct is not None and abs(spread_pct) >= spot_futures_spread:
            arrow = "⬆️" if spread_pct > 0 else "⬇️"
            alerts.append(f"Spread {spread_pct:+.2f}% {arrow}")
            other_signals += 1
            score += 1

        # === Распределение по категориям ===
        if (
            price_change <= -5.0 and
            rsi < rsi_low and
            funding <= -0.2 and
            oi_delta >= oi_delta_threshold and
            volatility >= 1.0
        ):
            return ("ANOMALY_LONG", symbol, score, alerts)

        if (
            price_change >= 5.0 and
            rsi > rsi_high and
            funding >= 0.2 and
            oi_delta >= oi_delta_threshold and
            volatility >= 1.0
        ):
            return ("ANOMALY_SHORT", symbol, score, alerts)

        if volume_spike and other_signals > 0:
            return ("VOLUME_PLUS_ANY", symbol, score, alerts)

        if score >= min_matches:
            return ("MIN_MATCHES", symbol, score, alerts)

        return None

    tasks = [analyze_symbol(symbol) for symbol in symbols]
    scanned = await asyncio.gather(*tasks)

    for result in scanned:
        if result:
            mode, symbol, score, alerts = result
            results[mode].append((symbol, score, alerts))

           # === Формируем отчёт с Markdown-ссылками и экранированием ===
    report = "📊 Сигналы по сценариям:\n"
    for mode, items in results.items():
        if not items:
            continue
        emoji = {
            "ANOMALY_LONG": "🟢 LONG",
            "ANOMALY_SHORT": "🔴 SHORT",
            "VOLUME_PLUS_ANY": "📊 ОБЪЁМ",
            "MIN_MATCHES": "✅ ПОДТВЕРЖДЕНО"
        }[mode]
        report += f"\n{emoji}:\n"
        for symbol, score, alerts in sorted(items, key=lambda x: x[1], reverse=True):
            link = f"{symbol} ({score}/8):\n"
            report += f"{link} ({score}/8):\n"
            for alert in alerts:
                report += f"• {alert}\n"

    if report.strip() == "📊 Сигналы по сценариям:":
        return "😴 Нет монет по выбранным условиям."

    return report

async def scan_pump_dump(price_threshold=5.0, volume_ratio_threshold=5.0, min_volume=10_000_000):
    """
    Показывает памп/дамп по цене ±5% и рост/падение объёма x5 на 1m,
    только если объём на 1m >= 1_000_000 (можешь менять).
    Если по какому-то индикатору нет монет — блок просто не показывается.
    Если вообще ничего нет — возвращает пустую строку.
    """
    symbols = get_all_symbols()
    pumps = []
    dumps = []
    volume_spikes = []
    volume_drops = []

    for symbol in symbols:
        try:
            d = await asyncio.wait_for(get_data(symbol, "1"), timeout=8)
            if not d:
                continue
            price_change = d.get("price_change")
            volume_ratio = d.get("volume_ratio")
            volume = d.get("volume", 0)
            try:
                volume = float(volume)
            except:
                volume = 0

            # Фильтр: только свечи с объёмом >= min_volume
            if volume < min_volume:
                continue

            # Памп/дамп по цене
            if price_change is not None and price_change >= price_threshold:
                pumps.append((symbol, price_change))
            elif price_change is not None and price_change <= -price_threshold:
                dumps.append((symbol, price_change))

            # Всплеск/дамп по объёму
            if volume_ratio is not None and volume_ratio >= volume_ratio_threshold:
                volume_spikes.append((symbol, volume_ratio))
            elif volume_ratio is not None and volume_ratio <= 1.0 / volume_ratio_threshold:
                volume_drops.append((symbol, volume_ratio))

        except Exception as e:
            logger.warning(f"⚠ Ошибка pump/dump сканирования {symbol}: {e}")

    result_lines = ["🚨 *Pump/Dump сканер (1m)*"]

    if pumps:
        result_lines.append("\n🔺 Памп по цене (>= +5%):")
        result_lines += [f"{s}: {pc:+.2f}%" for s, pc in pumps]
    if dumps:
        result_lines.append("\n🔻 Дамп по цене (<= -5%):")
        result_lines += [f"{s}: {pc:+.2f}%" for s, pc in dumps]
    if volume_spikes:
        result_lines.append("\n🔥 Всплеск объёма (x5 и более):")
        result_lines += [f"{s}: {v:.2f}x" for s, v in volume_spikes]
    if volume_drops:
        result_lines.append("\n💀 Дамп объёма (x0.2 и меньше):")
        result_lines += [f"{s}: {v:.2f}x" for s, v in volume_drops]

    if len(result_lines) == 1:
        return ""  # Вообще ничего не выводим если нет ни одного сигнала

    return "\n".join(result_lines)
    from collections import Counter

import aiohttp

async def find_simple_price_levels(
    symbol: str,
    interval: str = "60",     # 1H
    limit: int = 744,         # 1 месяц
    bounce_pct: float = 0.02, # минимальный % отскока
    level_tol: float = 0.001, # не используется, можно убрать
    min_bounces: int = 2,     # минимально отскоков на уровень
    round_digits: int = 4,    # точность округления
    look_ahead: int = 6,      # сколько свечей вперёд искать реакцию
    min_volume_mult: float = 2.0, # объём в свечке-отскоке в X раз выше медианного
):
    import aiohttp
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API error: {data.get('retMsg')}")
            ohlcv = data["result"]["list"]

    highs = [float(c[2]) for c in ohlcv]
    lows  = [float(c[3]) for c in ohlcv]
    closes = [float(c[4]) for c in ohlcv]
    volumes = [float(c[5]) for c in ohlcv]
    median_vol = sorted(volumes)[len(volumes)//2]

    # --- HIGH: сопротивления (отскок вниз)
    high_levels = {}
    for i in range(len(highs) - look_ahead):
        price = round(highs[i], round_digits)
        for j in range(1, look_ahead + 1):
            next_price = closes[i + j]
            next_vol = volumes[i + j]
            move = (next_price - price) / price
            # Ищем только сильные отскоки вниз
            if move > -bounce_pct or abs(move) < bounce_pct:
                continue
            if next_vol < min_volume_mult * median_vol:
                continue
            if price not in high_levels:
                high_levels[price] = []
            high_levels[price].append((move, next_vol))
            break

    # --- LOW: поддержки (отскок вверх)
    low_levels = {}
    for i in range(len(lows) - look_ahead):
        price = round(lows[i], round_digits)
        for j in range(1, look_ahead + 1):
            next_price = closes[i + j]
            next_vol = volumes[i + j]
            move = (next_price - price) / price
            # Ищем только сильные отскоки вверх
            if move < bounce_pct or abs(move) < bounce_pct:
                continue
            if next_vol < min_volume_mult * median_vol:
                continue
            if price not in low_levels:
                low_levels[price] = []
            low_levels[price].append((move, next_vol))
            break

    # --- Формируем итоговый результат
    result = []
    for lvl, bounces in high_levels.items():
        if len(bounces) >= min_bounces:
            avg_bounce = sum(abs(x[0]) for x in bounces) / len(bounces)
            avg_vol = sum(x[1] for x in bounces) / len(bounces)
            result.append((lvl, "high", len(bounces), avg_bounce * 100, avg_vol))
    for lvl, bounces in low_levels.items():
        if len(bounces) >= min_bounces:
            avg_bounce = sum(abs(x[0]) for x in bounces) / len(bounces)
            avg_vol = sum(x[1] for x in bounces) / len(bounces)
            result.append((lvl, "low", len(bounces), avg_bounce * 100, avg_vol))

    result.sort(key=lambda x: -x[2])  # по числу отскоков
    return result











    




        



# Остальные async-сканеры (`scan_rsi_extremes`, `scan_volume_spikes`, `scan_extreme_funding`, `scan_all_indicators`, `scan_pump_dump`) — не менялись, просто вставляй их сюда по необходимости.
# Везде убери дубли, мусор и следи за уникальностью функций.

# В конце, если это Telegram-бот:
# report = escape_markdown(report)
# client.send_message(..., report, parse_mode='MarkdownV2')

