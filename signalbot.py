import os
import re
import logging  # ← вот это добавь
import asyncio
from telethon import TelegramClient, events
from signals import find_simple_price_levels
from dotenv import load_dotenv
from signals import (
    get_rsi_signal,
    get_all_symbols,
    scan_rsi_extremes,
    scan_extreme_funding,
    scan_price_spikes,
    scan_volume_spikes,
    scan_all_indicators,
    scan_pump_dump,
    find_simple_price_levels
)
def escape_markdown(text: str) -> str:
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

BOT_USERNAME = "top5signalbot"  # <- твой username БЕЗ @

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
bot_token = os.getenv("BOT_TOKEN")

if not api_id or not api_hash or not bot_token:
    raise RuntimeError("❌ В .env должны быть API_ID, API_HASH и BOT_TOKEN")

try:
    api_id = int(api_id)
except ValueError:
    raise RuntimeError("❌ API_ID должен быть целым числом")

client = TelegramClient('signalbot', api_id, api_hash).start(bot_token=bot_token)
active_alerts = {}

async def safe_respond(event, text: str, buttons=None, linkify=True):
    try:
        if linkify:
            await event.respond(linkify_text(text), buttons=buttons, parse_mode='Markdown')
        else:
            await event.respond(text, buttons=buttons, parse_mode='Markdown')
    except Exception as e:
        print(f"safe_respond ERROR: {e}")

def make_token_link(symbol: str) -> str:
    return f"[{escape_markdown(symbol)}](https://t.me/{BOT_USERNAME}?start=signal_{symbol})"

def linkify_text(text):
    # Теперь ловит тикеры длиной от 1 буквы: A-Z, цифры, до 20 символов + USDT
    pattern = r'(?<!\w)([A-Z0-9]{1,20}USDT)(?!\w)'
    return re.sub(pattern, lambda m: make_token_link(m.group(1)), text)

def is_empty_pumpdump(msg: str) -> bool:
    """Проверяет, что все четыре блока результатов содержат только '- нет -'."""
    return (
        "🔺 Памп по цене" in msg and "- нет -" in msg and
        "🔻 Дамп по цене" in msg and msg.count("- нет -") >= 4
    )

@client.on(events.NewMessage(pattern=r'^/signal(?:\s+([A-Za-z0-9_]+))?$'))
async def cmd_signal(event):
    m = event.pattern_match
    symbol = m.group(1).upper() if m.group(1) else "BTCUSDT"
    valid = get_all_symbols()
    if valid is not None and symbol not in valid:
        await safe_respond(event, f"❌ Символ {symbol} не найден.", linkify=False)
        return

    sig, token_name = await get_rsi_signal(symbol)
    if not sig:
        await safe_respond(event, f"⚠️ Недостаточно данных по {symbol}.", linkify=False)
        return

    signal_link = f"https://t.me/{BOT_USERNAME}?start=signal_{symbol}"
    # Только одна ссылка вверху, никаких дублирующих BTCUSDT.P сверху!
    title = f"[{symbol}]({signal_link}) — RSI (1m / 5m / 15m)"

    await safe_respond(
        event,
        f"{title}\n\n{sig}\n\n`{token_name}`",
        linkify=False
    )


@client.on(events.NewMessage(pattern=r'^/rsi_scan$'))
async def cmd_rsi_scan(event):
    await safe_respond(event, "🔎 Сканирую RSI по 5m… ")
    result_text, symbols = await scan_rsi_extremes(return_symbols=True)
    await safe_respond(event, result_text)

@client.on(events.NewMessage(pattern=r'^/funding$'))
async def cmd_funding(event):
    await safe_respond(event, "📡 Сканирую аномальный фандинг…")
    result = await scan_extreme_funding()
    await safe_respond(event, result)

@client.on(events.NewMessage(pattern=r'^/movers$'))
async def cmd_movers(event):
    await safe_respond(event, "📈 Сканирую токены с изменением ±2% (5m)...")
    result = await scan_price_spikes()
    await safe_respond(event, result)

@client.on(events.NewMessage(pattern=r'^/volumes$'))
async def cmd_volumes(event):
    await safe_respond(event, "📊 Сканирую токены с объёмом x3+ (5m)...")
    result = await scan_volume_spikes()
    await safe_respond(event, result)

@client.on(events.NewMessage(pattern=r'^/pumpdump$'))
async def cmd_pumpdump(event):
    await safe_respond(event, "🚨 Сканирую пампы/дампы и объёмы (1m)…")
    result = await scan_pump_dump()
    if is_empty_pumpdump(result):
        await safe_respond(event, "❌ Нет свежих пампов или дампов.")
    else:
        await safe_respond(event, result)

@client.on(events.NewMessage(pattern=r'^/startpumpdump$'))
async def start_pumpdump_alert(event):
    chat_id = event.chat_id
    if f"pumpdump_{chat_id}" in active_alerts:
        await safe_respond(event, "⚠️ Pump/Dump-мониторинг уже запущен.")
    else:
        task = asyncio.create_task(run_pumpdump_alert_loop(chat_id))
        active_alerts[f"pumpdump_{chat_id}"] = task
        await safe_respond(event, "✅ Pump/Dump-мониторинг запущен, скриннер мониторит ±5% изменения цены и Х5 увеличение или уменьшение объемов на 1м свече, обновление каждую минуту.")

@client.on(events.NewMessage(pattern=r'^/stoppumpdump$'))
async def stop_pumpdump_alert(event):
    chat_id = event.chat_id
    task = active_alerts.pop(f"pumpdump_{chat_id}", None)
    if task:
        task.cancel()
        await safe_respond(event, "🛑 Pump/Dump-мониторинг остановлен.")
    else:
        await safe_respond(event, "ℹ️ Pump/Dump-мониторинг и так не активен.")

async def run_pumpdump_alert_loop(chat_id):
    try:
        while True:
            msg = await scan_pump_dump()
            msg = linkify_text(msg)
            if not msg.strip():      # <--- добавь вот эту строку!
                await asyncio.sleep(60)
                continue
            await client.send_message(chat_id, msg, parse_mode='Markdown')
            await asyncio.sleep(60)

    except asyncio.CancelledError:
        logger.info(f"PumpDump-мониторинг завершён для chat_id {chat_id}")
    except Exception as e:
        logger.warning(f"❌ Ошибка в run_pumpdump_alert_loop для {chat_id}: {e}")

@client.on(events.CallbackQuery(data=re.compile(b'^signal:(.+)$')))
async def callback_signal(event):
    symbol = event.data.decode().split(":")[1]
    await event.answer("⏱ Получаю RSI…", alert=False)
    sig, token_name = await get_rsi_signal(symbol)
    copy_hint = f"\n\n`{token_name}`\n_Нажми кнопку ниже для быстрого копирования тикера_"
    await safe_respond(event, f"{sig}{copy_hint}", linkify=False)

@client.on(events.NewMessage(pattern=r'^/startalert$'))
async def start_alert_command(event):
    chat_id = event.chat_id
    if chat_id in active_alerts:
        await safe_respond(event, "⚠️ Онлайн-мониторинг уже запущен.")
    else:
        task = asyncio.create_task(run_alert_loop(chat_id))
        active_alerts[chat_id] = task
        await safe_respond(event, "✅ Онлайн-мониторинг сигналов запущен.")

@client.on(events.NewMessage(pattern=r'^/stopalert$'))
async def stop_alert_command(event):
    chat_id = event.chat_id
    task = active_alerts.pop(chat_id, None)
    if task:
        task.cancel()
        await safe_respond(event, "🛑 Мониторинг сигналов остановлен.")
    else:
        await safe_respond(event, "ℹ️ Мониторинг и так не активен.")

async def run_alert_loop(chat_id):
    try:
        while True:
            msg = await scan_all_indicators()
            await client.send_message(chat_id, linkify_text(msg), parse_mode='Markdown')
            await asyncio.sleep(300)
    except asyncio.CancelledError:
        logger.info(f"✅ Мониторинг завершён для chat_id {chat_id}")
    except Exception as e:
        logger.warning(f"❌ Ошибка в run_alert_loop для {chat_id}: {e}")

@client.on(events.NewMessage(pattern=r'^/start(?:\s+signal_(\w+))?'))
async def handle_start_signal(event):
    m = event.pattern_match
    symbol = m.group(1)
    if not symbol:
        await safe_respond(event, "👋 Используй /signal \"Название токена (BTCUSDT)\" что бы увидеть информацию по индикаторам токена.", linkify=False)
        return

    symbol = symbol.upper()
    valid = get_all_symbols()
    if valid is not None and symbol not in valid:
        await safe_respond(event, f"❌ {symbol} не найден в списке ({len(valid)}).", linkify=False)
        return

    await safe_respond(event, f"📥 Получаю сигнал по {symbol}…", linkify=False)
    sig, token_name = await get_rsi_signal(symbol)

    # Кликабельный заголовок, как и в signal
    signal_link = f"https://t.me/{BOT_USERNAME}?start=signal_{symbol}"
    title = f"[{symbol}]({signal_link}) — RSI (1m / 5m / 15m)"

    await safe_respond(
        event,
        f"{title}\n\n{sig}\n\n`{token_name}`",
        linkify=False
    )
@client.on(events.NewMessage(pattern=r'^/levels$'))
async def cmd_levels(event):
    max_tokens = 10
    await safe_respond(event, "📊 Сканирую монеты, ищу уровни с реальными отскоками (1H, 1 месяц)...")
    symbols = get_all_symbols()
    limit = 150  # 788 это 1 месяц по 1H

    # Твои фильтры:
    bounce_pct = 0.05       # 5% отскок
    min_bounces = 3         # минимум 2 отскока
    min_volume_mult = 2.0  # объем X10 к медиане
    direction = "resist"    # или "support"

    async def process(symbol):
        try:
            levels = await find_simple_price_levels(
                symbol,
                interval="240",
                limit=limit,
                bounce_pct=bounce_pct,
                min_bounces=min_bounces,
                round_digits=4,
                look_ahead=6,
                min_volume_mult=min_volume_mult,
            )
            if not levels:
                return None
            # top_level = (level, lvl_type, hits, avg_pct, avg_vol)
            top_level = levels[0]
            return symbol, *top_level
        except Exception as e:
            logger.warning(f"Ошибка анализа {symbol}: {e}")
            return None

    tasks = [process(s) for s in symbols]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r]

    # Сортируем по числу отскоков ↓↓↓
    results.sort(key=lambda x: -x[3])

    if not results:
        await safe_respond(event, "❌ Нет уровней с сильными отскоками.")
        return

    msg = "📊 *Топ-10 монет с сильными отскоками (1H, 1 месяц):*\n\n"
    for symbol, level, lvl_type, hits, avg_pct, avg_vol in results:
        price_fmt = f"{level:.5f}" if level < 1 else f"{level:.2f}"
        tag = "HIGH" if lvl_type == "high" else "LOW"
        msg += (
            f"{symbol} {tag} {price_fmt} — {hits} отскоков, "
            f"ср.отскок {avg_pct:.2f}%, ср.объём {int(avg_vol):,}\n"
        )

    await safe_respond(event, msg)






    
if __name__ == "__main__":
    print("📡 Бот запущен")
    client.run_until_disconnected()
    




