#!/usr/bin/env python3
"""
回测引擎 - 执行大规模回测

功能：
1. 加载历史数据
2. 运行自适应策略
3. 模拟交易
4. 记录详细结果
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    return_pct: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'timeout'
    signal_score: float
    volume_percentile: float
    bsr_percentile: float
    net_buy_percentile: float
    holding_minutes: int
    max_profit_pct: float
    max_drawdown_pct: float


@dataclass
class DailyStats:
    """每日统计"""
    date: str
    symbols_count: int
    signals_count: int
    trades_count: int
    win_count: int
    loss_count: int
    total_return_pct: float
    avg_return_pct: float
    max_return_pct: float
    min_return_pct: float
    win_rate: float


class AdaptiveStrategy:
    """自适应策略"""

    def __init__(self, config: dict = STRATEGY_CONFIG):
        self.lookback_window = config['lookback_window']
        self.score_threshold = config['score_threshold']
        self.price_position_threshold = config['price_position_threshold']
        self.volume_weight = config['volume_weight']
        self.bsr_weight = config['buy_sell_ratio_weight']
        self.net_buy_weight = config['net_buy_weight']

    def calc_percentile(self, value: float, history: np.ndarray) -> float:
        """计算百分位"""
        if len(history) < 5:
            return 0.5
        return np.sum(history < value) / len(history)

    def process_minute_data(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """处理分钟数据并生成信号"""
        df = minute_df.copy()

        # 计算衍生指标
        df['net_buy'] = df['buy_volume'] - df['sell_volume']
        df['buy_sell_ratio'] = df['buy_volume'] / df['sell_volume'].replace(0, 0.001)

        # 计算滚动百分位
        window = self.lookback_window

        def rolling_percentile(series):
            result = []
            for i in range(len(series)):
                if i < window:
                    result.append(0.5)
                else:
                    hist = series.iloc[i-window:i].values
                    curr = series.iloc[i]
                    pct = np.sum(hist < curr) / len(hist)
                    result.append(pct)
            return result

        df['volume_pct'] = rolling_percentile(df['volume'])
        df['bsr_pct'] = rolling_percentile(df['buy_sell_ratio'])
        df['net_buy_pct'] = rolling_percentile(df['net_buy'])

        # 价格位置
        df['price_high_roll'] = df['high'].rolling(window, min_periods=1).max()
        df['price_low_roll'] = df['low'].rolling(window, min_periods=1).min()
        df['price_range'] = df['price_high_roll'] - df['price_low_roll']
        df['price_position'] = (df['close'] - df['price_low_roll']) / df['price_range'].replace(0, 1)

        # 综合评分
        df['signal_score'] = (
            df['volume_pct'] * self.volume_weight +
            df['bsr_pct'] * self.bsr_weight +
            df['net_buy_pct'] * self.net_buy_weight
        )

        # 信号条件
        df['is_signal'] = (
            (df['signal_score'] >= self.score_threshold) &
            (df['price_position'] >= self.price_position_threshold)
        )

        return df


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        strategy: AdaptiveStrategy = None,
        trade_config: dict = TRADE_CONFIG
    ):
        self.strategy = strategy or AdaptiveStrategy()
        self.take_profit = trade_config['take_profit_pct']
        self.stop_loss = trade_config['stop_loss_pct']
        self.max_hold = trade_config['max_hold_minutes']
        self.cooldown = trade_config['signal_cooldown_minutes']
        self.position_size = trade_config['position_size_usdt']
        self.slippage = trade_config['slippage_pct']
        self.fee = trade_config['fee_pct']

        self.trades: List[Trade] = []
        self.daily_stats: List[DailyStats] = []
        self.signals_detail = []  # 所有信号详情

    def load_agg_trades(self, file_path: Path) -> Optional[pd.DataFrame]:
        """加载aggTrades数据并转换为分钟K线"""
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return None

            # 转换时间
            df['datetime'] = pd.to_datetime(df['transact_time'], unit='ms')
            df['minute'] = df['datetime'].dt.floor('1min')

            # 按分钟聚合
            agg = df.groupby('minute').agg({
                'price': ['min', 'max', 'first', 'last'],
                'quantity': 'sum',
                'is_buyer_maker': lambda x: (
                    df.loc[x.index, 'quantity'] * (1 - df.loc[x.index, 'is_buyer_maker'].map({True: 1, False: 0, 'true': 1, 'false': 0}))
                ).sum()
            }).reset_index()

            agg.columns = ['minute', 'low', 'high', 'open', 'close', 'volume', 'buy_volume']
            agg['sell_volume'] = agg['volume'] - agg['buy_volume']

            return agg

        except Exception as e:
            print(f"  加载数据失败 {file_path}: {e}")
            return None

    def simulate_trade(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        symbol: str
    ) -> Optional[Trade]:
        """模拟单笔交易"""
        entry_row = df.iloc[signal_idx]
        entry_price = entry_row['close'] * (1 + self.slippage / 100)  # 考虑滑点
        entry_time = entry_row['minute']

        exit_price = None
        exit_reason = None
        exit_time = None
        max_profit = 0
        max_drawdown = 0

        for hold_min in range(1, self.max_hold + 1):
            future_idx = signal_idx + hold_min
            if future_idx >= len(df):
                break

            bar = df.iloc[future_idx]
            high_ret = (bar['high'] - entry_price) / entry_price * 100
            low_ret = (bar['low'] - entry_price) / entry_price * 100

            max_profit = max(max_profit, high_ret)
            max_drawdown = min(max_drawdown, low_ret)

            # 检查止盈
            if high_ret >= self.take_profit:
                exit_price = entry_price * (1 + self.take_profit / 100)
                exit_reason = 'take_profit'
                exit_time = bar['minute']
                break

            # 检查止损
            if low_ret <= self.stop_loss:
                exit_price = entry_price * (1 + self.stop_loss / 100)
                exit_reason = 'stop_loss'
                exit_time = bar['minute']
                break

        # 超时平仓
        if exit_price is None:
            final_idx = min(signal_idx + self.max_hold, len(df) - 1)
            final_bar = df.iloc[final_idx]
            exit_price = final_bar['close']
            exit_reason = 'timeout'
            exit_time = final_bar['minute']

        # 计算收益（扣除手续费）
        gross_return = (exit_price - entry_price) / entry_price * 100
        net_return = gross_return - self.fee * 2  # 开仓+平仓手续费

        holding_minutes = (exit_time - entry_time).total_seconds() / 60 if exit_time else self.max_hold

        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            return_pct=net_return,
            exit_reason=exit_reason,
            signal_score=entry_row['signal_score'],
            volume_percentile=entry_row['volume_pct'],
            bsr_percentile=entry_row['bsr_pct'],
            net_buy_percentile=entry_row['net_buy_pct'],
            holding_minutes=int(holding_minutes),
            max_profit_pct=max_profit,
            max_drawdown_pct=max_drawdown
        )

    def backtest_symbol_day(
        self,
        symbol: str,
        file_path: Path,
        date: str
    ) -> List[Trade]:
        """回测单个合约单天"""
        # 加载数据
        minute_df = self.load_agg_trades(file_path)
        if minute_df is None or len(minute_df) < self.strategy.lookback_window * 2:
            return []

        # 生成信号
        df = self.strategy.process_minute_data(minute_df)

        # 获取信号点
        signal_indices = df[df['is_signal']].index.tolist()

        trades = []
        last_trade_end = -self.cooldown

        for idx in signal_indices:
            # 冷却检查
            if idx < last_trade_end + self.cooldown:
                continue

            # 记录信号详情
            row = df.iloc[idx]
            self.signals_detail.append({
                'date': date,
                'symbol': symbol,
                'time': str(row['minute']),
                'price': row['close'],
                'signal_score': row['signal_score'],
                'volume_pct': row['volume_pct'],
                'bsr_pct': row['bsr_pct'],
                'net_buy_pct': row['net_buy_pct'],
            })

            # 模拟交易
            trade = self.simulate_trade(df, idx, symbol)
            if trade:
                trades.append(trade)
                last_trade_end = idx + trade.holding_minutes

        return trades

    def run_backtest(self, data_dir: Path, daily_symbols: Dict[str, List[str]]):
        """运行完整回测"""
        print("\n" + "=" * 70)
        print("开始回测")
        print("=" * 70)

        all_trades = []

        for date, symbols in sorted(daily_symbols.items()):
            print(f"\n日期: {date}, 合约数: {len(symbols)}")

            daily_trades = []

            for symbol in symbols:
                file_path = data_dir / symbol / f"{symbol}-aggTrades-{date}.csv"
                if not file_path.exists():
                    continue

                trades = self.backtest_symbol_day(symbol, file_path, date)
                daily_trades.extend(trades)

            all_trades.extend(daily_trades)

            # 计算每日统计
            if daily_trades:
                wins = [t for t in daily_trades if t.return_pct > 0]
                losses = [t for t in daily_trades if t.return_pct <= 0]

                stats = DailyStats(
                    date=date,
                    symbols_count=len(symbols),
                    signals_count=len([s for s in self.signals_detail if s['date'] == date]),
                    trades_count=len(daily_trades),
                    win_count=len(wins),
                    loss_count=len(losses),
                    total_return_pct=sum(t.return_pct for t in daily_trades),
                    avg_return_pct=np.mean([t.return_pct for t in daily_trades]),
                    max_return_pct=max(t.return_pct for t in daily_trades),
                    min_return_pct=min(t.return_pct for t in daily_trades),
                    win_rate=len(wins) / len(daily_trades) * 100 if daily_trades else 0
                )
                self.daily_stats.append(stats)

                print(f"  交易: {len(daily_trades)}, 胜率: {stats.win_rate:.1f}%, "
                      f"收益: {stats.total_return_pct:.2f}%")

        self.trades = all_trades
        return all_trades

    def get_results(self) -> Dict:
        """获取回测结果"""
        if not self.trades:
            return {'error': 'No trades'}

        trades_df = pd.DataFrame([asdict(t) for t in self.trades])

        # 总体统计
        total_trades = len(self.trades)
        wins = trades_df[trades_df['return_pct'] > 0]
        losses = trades_df[trades_df['return_pct'] <= 0]

        results = {
            'summary': {
                'total_trades': total_trades,
                'win_trades': len(wins),
                'loss_trades': len(losses),
                'win_rate': len(wins) / total_trades * 100,
                'total_return_pct': trades_df['return_pct'].sum(),
                'avg_return_pct': trades_df['return_pct'].mean(),
                'std_return_pct': trades_df['return_pct'].std(),
                'max_return_pct': trades_df['return_pct'].max(),
                'min_return_pct': trades_df['return_pct'].min(),
                'avg_win_pct': wins['return_pct'].mean() if len(wins) > 0 else 0,
                'avg_loss_pct': losses['return_pct'].mean() if len(losses) > 0 else 0,
                'profit_factor': abs(wins['return_pct'].sum() / losses['return_pct'].sum()) if len(losses) > 0 and losses['return_pct'].sum() != 0 else float('inf'),
                'sharpe_ratio': trades_df['return_pct'].mean() / trades_df['return_pct'].std() * np.sqrt(252) if trades_df['return_pct'].std() > 0 else 0,
            },
            'by_exit_reason': trades_df.groupby('exit_reason').agg({
                'return_pct': ['count', 'mean', 'sum']
            }).to_dict(),
            'by_symbol': trades_df.groupby('symbol').agg({
                'return_pct': ['count', 'mean', 'sum']
            }).to_dict(),
            'daily_stats': [asdict(s) for s in self.daily_stats],
            'trades': trades_df.to_dict('records'),
            'signals': self.signals_detail,
        }

        return results


def main():
    """主程序"""
    data_dir = Path(DATA_DIR)

    # 加载筛选结果
    symbols_file = data_dir / 'daily_symbols.json'
    if not symbols_file.exists():
        print("请先运行 data_downloader.py 下载数据")
        return

    with open(symbols_file) as f:
        daily_symbols = json.load(f)

    print(f"加载数据: {len(daily_symbols)} 天")

    # 创建回测引擎
    engine = BacktestEngine()

    # 运行回测
    engine.run_backtest(data_dir, daily_symbols)

    # 获取结果
    results = engine.get_results()

    # 保存结果
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存JSON结果
    with open(output_dir / f'backtest_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 打印摘要
    print("\n" + "=" * 70)
    print("回测结果摘要")
    print("=" * 70)
    summary = results['summary']
    print(f"总交易次数: {summary['total_trades']}")
    print(f"胜率: {summary['win_rate']:.1f}%")
    print(f"总收益: {summary['total_return_pct']:.2f}%")
    print(f"平均收益: {summary['avg_return_pct']:.2f}%")
    print(f"平均盈利: {summary['avg_win_pct']:.2f}%")
    print(f"平均亏损: {summary['avg_loss_pct']:.2f}%")
    print(f"盈亏比: {summary['profit_factor']:.2f}")
    print(f"夏普比率: {summary['sharpe_ratio']:.2f}")

    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
