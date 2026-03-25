"""
大规模回测框架配置文件
"""

from datetime import datetime, timedelta

# ============================================================
# 回测时间范围
# ============================================================
BACKTEST_DAYS = 30  # 回测天数
END_DATE = datetime(2026, 3, 20)  # 结束日期
START_DATE = END_DATE - timedelta(days=BACKTEST_DAYS)

# ============================================================
# 合约筛选条件
# ============================================================
MIN_DAILY_VOLUME_USDT = 1_000_000    # 最小日交易量 100万U
MAX_DAILY_VOLUME_USDT = 10_000_000   # 最大日交易量 1000万U

# 备选范围（如果目标范围合约太多，可以缩小范围）
# MIN_DAILY_VOLUME_USDT = 3_000_000    # 300万U
# MAX_DAILY_VOLUME_USDT = 8_000_000    # 800万U

# ============================================================
# 策略参数
# ============================================================
STRATEGY_CONFIG = {
    'lookback_window': 60,          # 回看窗口（分钟）
    'score_threshold': 0.95,        # 综合评分阈值
    'price_position_threshold': 0.8, # 价格位置阈值
    'volume_weight': 0.3,           # 成交量权重
    'buy_sell_ratio_weight': 0.4,   # 买卖比权重
    'net_buy_weight': 0.3,          # 净买入权重
}

# ============================================================
# 交易参数
# ============================================================
TRADE_CONFIG = {
    'take_profit_pct': 3.0,         # 止盈比例 %
    'stop_loss_pct': -3.0,          # 止损比例 %
    'max_hold_minutes': 30,         # 最大持仓时间（分钟）
    'signal_cooldown_minutes': 30,  # 同一币种信号冷却时间
    'position_size_usdt': 10000,    # 每次交易仓位（USDT）
    'slippage_pct': 0.1,            # 滑点假设 %
    'fee_pct': 0.04,                # 手续费 % (maker+taker)
}

# ============================================================
# 数据源配置
# ============================================================
BINANCE_DATA_URL = "https://data.binance.vision/data/futures/um/daily"
DATA_TYPES = ['aggTrades', 'metrics']  # 需要下载的数据类型

# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = "backtest/results"
LOG_DIR = "backtest/logs"
DATA_DIR = "backtest/data"
