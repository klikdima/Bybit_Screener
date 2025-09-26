# screener_config.py

# --- Пороговые значения индикаторов ---
RSI_LOW = 30
RSI_HIGH = 70

FUNDING_THRESHOLD = 0.9        # Funding rate (в процентах)
PRICE_THRESHOLD = 5.0           # Изменение цены за 5 минут (%)
VOLUME_RATIO_THRESHOLD = 4.0    # Объём в x раз выше среднего
OI_DELTA_THRESHOLD = 1.0        # Изменение open interest (%)
IMBALANCE_THRESHOLD = 70.0      # Доля buy/sell в стакане (%)
VOLATILITY_THRESHOLD = 0.5      # Волатильность (%)
SPOT_FUTURES_SPREAD = 0.5       # Разница между спотом и фьючами (%)

# --- Кастомный режим работы скриннера ---
# Доступные варианты:
# "VOLUME_PLUS_ANY", "MIN_MATCHES", "VOLUME_PLUS_PRICE_RSI_PLUS_ANY", "ALL"
CUSTOM_MODE = "ALL"

# --- Минимум совпадений (для режима MIN_MATCHES) ---
MIN_MATCHES = 3
