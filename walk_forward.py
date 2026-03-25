#!/usr/bin/env python3
"""
滚动前向验证系统（Walk-Forward Optimization）

核心思路：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         滚动前向验证                                          │
│                                                                             │
│  时间线: ──────────────────────────────────────────────────────────────────▶ │
│                                                                             │
│  第1轮:  [===训练期===]  [==验证期==]                                         │
│          Jan 1-10       Jan 11-20                                           │
│                                                                             │
│  第2轮:  [======训练期======]  [==验证期==]                                   │
│          Jan 1-20 (累积)      Jan 21-30                                     │
│                                                                             │
│  第3轮:  [=========训练期=========]  [==验证期==]                             │
│          Jan 1-30 (累积)            Jan 31 - Feb 9                          │
│                                                                             │
│  特点：                                                                      │
│  1. 永远用过去的数据训练，预测未来 → 无穿越                                    │
│  2. 训练数据累积增长 → 模型越来越稳健                                         │
│  3. 保留学习到的参数 → 知识积累                                               │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from self_evolving import FeatureEngine, TradeRecord, EvolutionState


@dataclass
class PeriodResult:
    """单周期结果"""
    period: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # 训练指标
    train_samples: int = 0
    train_precision: float = 0.0
    # 验证指标（真实前向测试）
    test_trades: int = 0
    test_win_rate: float = 0.0
    test_return: float = 0.0
    test_avg_return: float = 0.0
    # 模型参数
    threshold: float = 0.5
    blacklist_hours: List[int] = field(default_factory=list)
    blacklist_symbols: List[str] = field(default_factory=list)


class WalkForwardSystem:
    """滚动前向验证系统"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'windows': [15, 30, 60],
            'target_profit': 3.0,
            'stop_loss': -3.0,
            'hold_minutes': 30,
            'min_precision': 0.6,
            'train_days': 10,      # 每个训练周期天数
            'test_days': 10,       # 每个验证周期天数
            'evolution_rounds': 3,  # 每个周期内进化轮数
        }

        self.feature_engine = FeatureEngine(self.config['windows'])
        self.model = None
        self.threshold = 0.5
        self.state = EvolutionState()

        # 累积数据
        self.all_train_data = None
        self.period_results: List[PeriodResult] = []

    def load_data_for_date(self, date_str: str, data_dir: Path) -> Optional[pd.DataFrame]:
        """加载指定日期的数据"""
        # 尝试多种路径
        patterns = [
            data_dir / f'*-aggTrades-{date_str}.csv',
            data_dir / 'data' / f'**/*-aggTrades-{date_str}.csv',
        ]

        files = []
        for pattern in patterns:
            files.extend(list(pattern.parent.glob(pattern.name)))

        if not files:
            return None

        all_data = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
                df['datetime'] = pd.to_datetime(df['transact_time'], unit='ms')
                df['minute'] = df['datetime'].dt.floor('1min')

                symbol = fp.stem.split('-')[0]

                agg = df.groupby('minute').agg({
                    'price': ['min', 'max', 'first', 'last'],
                    'quantity': 'sum',
                    'is_buyer_maker': lambda x: (
                        df.loc[x.index, 'quantity'] *
                        (1 - df.loc[x.index, 'is_buyer_maker'].map(
                            {True: 1, False: 0, 'true': 1, 'false': 0}
                        ))
                    ).sum()
                }).reset_index()

                agg.columns = ['minute', 'low', 'high', 'open', 'close', 'volume', 'buy_volume']
                agg['sell_volume'] = agg['volume'] - agg['buy_volume']
                agg['symbol'] = symbol
                agg['hour'] = agg['minute'].dt.hour
                agg['date'] = date_str

                all_data.append(agg)
            except Exception as e:
                print(f"    警告: 加载 {fp} 失败: {e}")

        if not all_data:
            return None

        return pd.concat(all_data, ignore_index=True)

    def load_data_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        data_dir: Path
    ) -> pd.DataFrame:
        """加载一段时间的数据"""
        all_data = []
        current = start_date

        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            df = self.load_data_for_date(date_str, data_dir)
            if df is not None:
                all_data.append(df)
            current += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """生成标签"""
        labels = []
        tp = self.config['target_profit']
        sl = self.config['stop_loss']
        hold = self.config['hold_minutes']

        for i in range(len(df)):
            if i + hold >= len(df):
                labels.append(np.nan)
                continue

            if df.iloc[i]['symbol'] != df.iloc[i + hold]['symbol']:
                labels.append(np.nan)
                continue

            entry = df.iloc[i]['close']
            label = 0

            for j in range(1, hold + 1):
                if i + j >= len(df):
                    break
                bar = df.iloc[i + j]
                high_ret = (bar['high'] - entry) / entry * 100
                low_ret = (bar['low'] - entry) / entry * 100

                if high_ret >= tp:
                    label = 1
                    break
                if low_ret <= sl:
                    label = 0
                    break

            labels.append(label)

        return pd.Series(labels, index=df.index)

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        warm_start: bool = False
    ) -> Dict:
        """
        训练模型

        Args:
            warm_start: 是否在现有模型基础上继续训练（累积学习）
        """
        if not HAS_LGB:
            raise ImportError("需要安装lightgbm")

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        # 时间序列分割
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # 累积学习：在现有模型基础上继续训练
        if warm_start and self.model is not None:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=100,  # 增量训练较少轮数
                valid_sets=[val_data],
                init_model=self.model,  # 关键：从现有模型继续
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

        # 找最优阈值
        val_pred = self.model.predict(X_val)
        y_val_np = y_val.values

        best_threshold = 0.5
        best_score = -float('inf')

        for th in np.arange(0.3, 0.8, 0.02):
            pred = (val_pred >= th).astype(int)
            tp = ((pred == 1) & (y_val_np == 1)).sum()
            fp = ((pred == 1) & (y_val_np == 0)).sum()
            precision = tp / max(tp + fp, 1)

            score = precision * np.sqrt(tp + fp) if precision >= self.config['min_precision'] else 0

            if score > best_score:
                best_score = score
                best_threshold = th

        self.threshold = best_threshold

        # 特征重要性
        importance = dict(zip(
            self.feature_engine.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))
        self.state.feature_importance = importance

        # 验证指标
        val_pred_binary = (val_pred >= best_threshold).astype(int)
        tp = ((val_pred_binary == 1) & (y_val_np == 1)).sum()
        fp = ((val_pred_binary == 1) & (y_val_np == 0)).sum()
        fn = ((val_pred_binary == 0) & (y_val_np == 1)).sum()

        return {
            'threshold': best_threshold,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'n_signals': int(val_pred_binary.sum()),
            'n_samples': len(X),
        }

    def backtest(self, df: pd.DataFrame, features: List[str]) -> List[TradeRecord]:
        """执行回测"""
        if self.model is None:
            return []

        trades = []
        X = df[features]

        valid_mask = ~X.isna().any(axis=1)
        predictions = np.zeros(len(df))
        predictions[valid_mask] = self.model.predict(X[valid_mask])

        signals = predictions >= self.threshold

        # 应用过滤
        if self.state.blacklist_symbols:
            symbol_mask = ~df['symbol'].isin(self.state.blacklist_symbols)
            signals = signals & symbol_mask.values

        if self.state.blacklist_hours:
            hour_mask = ~df['hour'].isin(self.state.blacklist_hours)
            signals = signals & hour_mask.values

        # 模拟交易
        from collections import defaultdict
        symbol_cooldown = defaultdict(int)
        cooldown = self.config['hold_minutes']

        for i in np.where(signals)[0]:
            symbol = df.iloc[i]['symbol']

            if i < symbol_cooldown[symbol]:
                continue

            if i + self.config['hold_minutes'] >= len(df):
                continue

            if df.iloc[i + self.config['hold_minutes']]['symbol'] != symbol:
                continue

            entry_price = df.iloc[i]['close'] * 1.001

            exit_price = None
            exit_reason = 'timeout'

            for h in range(1, self.config['hold_minutes'] + 1):
                bar = df.iloc[i + h]
                high_ret = (bar['high'] - entry_price) / entry_price * 100
                low_ret = (bar['low'] - entry_price) / entry_price * 100

                if high_ret >= self.config['target_profit']:
                    exit_price = entry_price * (1 + self.config['target_profit'] / 100)
                    exit_reason = 'take_profit'
                    break
                if low_ret <= self.config['stop_loss']:
                    exit_price = entry_price * (1 + self.config['stop_loss'] / 100)
                    exit_reason = 'stop_loss'
                    break

            if exit_price is None:
                exit_price = df.iloc[i + self.config['hold_minutes']]['close']

            ret = (exit_price - entry_price) / entry_price * 100 - 0.08

            trades.append(TradeRecord(
                symbol=symbol,
                entry_time=str(df.iloc[i]['minute']),
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret,
                exit_reason=exit_reason,
                signal_proba=predictions[i],
                hour=df.iloc[i]['hour']
            ))

            symbol_cooldown[symbol] = i + cooldown

        return trades

    def evolve_on_trades(self, trades: List[TradeRecord]):
        """根据交易结果进化参数"""
        if not trades:
            return

        trades_df = pd.DataFrame([asdict(t) for t in trades])

        # 分析时段表现
        hour_stats = trades_df.groupby('hour')['return_pct'].agg(['count', 'mean'])
        bad_hours = hour_stats[hour_stats['mean'] < -0.5].index.tolist()

        for h in bad_hours:
            if h not in self.state.blacklist_hours:
                self.state.blacklist_hours.append(h)

        # 分析币种表现
        symbol_stats = trades_df.groupby('symbol')['return_pct'].agg(['count', 'mean'])
        bad_symbols = symbol_stats[(symbol_stats['count'] >= 3) & (symbol_stats['mean'] < -0.5)].index.tolist()

        for s in bad_symbols:
            if s not in self.state.blacklist_symbols:
                self.state.blacklist_symbols.append(s)

        # 调整阈值
        precision = (trades_df['return_pct'] > 0).mean()
        if precision < self.config['min_precision']:
            self.threshold = min(self.threshold + 0.03, 0.8)

    def run_walk_forward(
        self,
        data_dir: str,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        运行滚动前向验证

        Args:
            data_dir: 数据目录
            start_date: 起始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        """
        data_path = Path(data_dir)
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        train_days = self.config['train_days']
        test_days = self.config['test_days']

        print("=" * 70)
        print("滚动前向验证系统")
        print("=" * 70)
        print(f"  数据目录: {data_dir}")
        print(f"  总时间范围: {start_date} 至 {end_date}")
        print(f"  训练周期: {train_days} 天")
        print(f"  验证周期: {test_days} 天")
        print("=" * 70)

        period = 0
        train_start = start
        cumulative_train_data = None

        while True:
            period += 1
            train_end = train_start + timedelta(days=train_days * period - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days - 1)

            # 检查是否超出范围
            if test_end > end:
                print(f"\n验证期 {test_end.strftime('%Y-%m-%d')} 超出范围，停止")
                break

            print(f"\n{'='*70}")
            print(f"第 {period} 轮")
            print(f"{'='*70}")
            print(f"  训练期: {train_start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')}")
            print(f"  验证期: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')}")

            # 加载新增训练数据
            if period == 1:
                new_train_data = self.load_data_for_period(train_start, train_end, data_path)
            else:
                # 只加载新增部分
                prev_train_end = train_start + timedelta(days=train_days * (period - 1) - 1)
                new_train_data = self.load_data_for_period(
                    prev_train_end + timedelta(days=1),
                    train_end,
                    data_path
                )

            if new_train_data.empty:
                print(f"  警告: 无训练数据，跳过此轮")
                continue

            # 累积训练数据
            if cumulative_train_data is None:
                cumulative_train_data = new_train_data
            else:
                cumulative_train_data = pd.concat(
                    [cumulative_train_data, new_train_data],
                    ignore_index=True
                )

            print(f"\n[训练数据]")
            print(f"  累积样本: {len(cumulative_train_data)}")
            print(f"  涉及币种: {cumulative_train_data['symbol'].nunique()}")

            # 计算特征
            train_featured = self.feature_engine.compute(cumulative_train_data)
            train_featured['label'] = self.generate_labels(train_featured)
            train_valid = train_featured.dropna(subset=['label'] + self.feature_engine.feature_names)

            if len(train_valid) < 100:
                print(f"  有效样本不足 ({len(train_valid)}), 跳过")
                continue

            X_train = train_valid[self.feature_engine.feature_names]
            y_train = train_valid['label'].astype(int)

            # 训练（累积学习）
            print(f"\n[模型训练]")
            warm_start = period > 1  # 第一轮从头训练，之后累积学习
            train_result = self.train_model(X_train, y_train, warm_start=warm_start)
            print(f"  阈值: {train_result['threshold']:.2f}")
            print(f"  精确率: {train_result['precision']:.2%}")
            print(f"  召回率: {train_result['recall']:.2%}")

            # 内部进化（在训练数据上）
            print(f"\n[内部进化]")
            for evo_round in range(self.config['evolution_rounds']):
                train_trades = self.backtest(train_valid, self.feature_engine.feature_names)
                if train_trades:
                    self.evolve_on_trades(train_trades)

            if self.state.blacklist_hours:
                print(f"  黑名单时段: {self.state.blacklist_hours}")
            if self.state.blacklist_symbols:
                print(f"  黑名单币种: {self.state.blacklist_symbols[:5]}...")

            # 加载验证数据
            test_data = self.load_data_for_period(test_start, test_end, data_path)

            if test_data.empty:
                print(f"\n[前向验证] 无验证数据")
                result = PeriodResult(
                    period=period,
                    train_start=train_start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_samples=len(train_valid),
                    train_precision=train_result['precision'],
                    threshold=self.threshold,
                    blacklist_hours=self.state.blacklist_hours.copy(),
                    blacklist_symbols=self.state.blacklist_symbols.copy(),
                )
                self.period_results.append(result)
                continue

            # 计算验证数据特征
            test_featured = self.feature_engine.compute(test_data)
            test_valid = test_featured.dropna(subset=self.feature_engine.feature_names)

            # 前向验证（真实测试）
            print(f"\n[前向验证] - 真实样本外测试")
            test_trades = self.backtest(test_valid, self.feature_engine.feature_names)

            if test_trades:
                wins = sum(1 for t in test_trades if t.return_pct > 0)
                total_ret = sum(t.return_pct for t in test_trades)
                win_rate = wins / len(test_trades)
                avg_ret = total_ret / len(test_trades)

                print(f"  交易数: {len(test_trades)}")
                print(f"  胜率: {win_rate*100:.1f}%")
                print(f"  总收益: {total_ret:.2f}%")
                print(f"  平均收益: {avg_ret:.2f}%")

                result = PeriodResult(
                    period=period,
                    train_start=train_start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_samples=len(train_valid),
                    train_precision=train_result['precision'],
                    test_trades=len(test_trades),
                    test_win_rate=win_rate,
                    test_return=total_ret,
                    test_avg_return=avg_ret,
                    threshold=self.threshold,
                    blacklist_hours=self.state.blacklist_hours.copy(),
                    blacklist_symbols=self.state.blacklist_symbols.copy(),
                )

                # 用验证期数据继续进化（为下一轮准备）
                self.evolve_on_trades(test_trades)
            else:
                print(f"  无交易信号")
                result = PeriodResult(
                    period=period,
                    train_start=train_start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_samples=len(train_valid),
                    train_precision=train_result['precision'],
                    threshold=self.threshold,
                    blacklist_hours=self.state.blacklist_hours.copy(),
                    blacklist_symbols=self.state.blacklist_symbols.copy(),
                )

            self.period_results.append(result)

        # 汇总结果
        self._print_summary()

        return {
            'period_results': [asdict(r) for r in self.period_results],
            'final_state': self.state.to_dict(),
            'config': self.config,
        }

    def _print_summary(self):
        """打印汇总结果"""
        print(f"\n{'='*70}")
        print("滚动前向验证汇总")
        print(f"{'='*70}")

        if not self.period_results:
            print("无验证结果")
            return

        # 表格输出
        print(f"\n{'轮次':<6}{'训练期':<24}{'验证期':<24}{'交易':<6}{'胜率':<8}{'收益':<10}")
        print("-" * 78)

        total_trades = 0
        total_wins = 0
        total_return = 0

        for r in self.period_results:
            train_period = f"{r.train_start} ~ {r.train_end}"
            test_period = f"{r.test_start} ~ {r.test_end}"

            if r.test_trades > 0:
                print(f"{r.period:<6}{train_period:<24}{test_period:<24}"
                      f"{r.test_trades:<6}{r.test_win_rate*100:>5.1f}%  {r.test_return:>+7.2f}%")
                total_trades += r.test_trades
                total_wins += int(r.test_trades * r.test_win_rate)
                total_return += r.test_return
            else:
                print(f"{r.period:<6}{train_period:<24}{test_period:<24}{'0':<6}{'N/A':<8}{'N/A':<10}")

        print("-" * 78)

        if total_trades > 0:
            overall_win_rate = total_wins / total_trades
            print(f"{'总计':<6}{'':<24}{'':<24}"
                  f"{total_trades:<6}{overall_win_rate*100:>5.1f}%  {total_return:>+7.2f}%")

            print(f"\n关键指标:")
            print(f"  - 总交易次数: {total_trades}")
            print(f"  - 整体胜率: {overall_win_rate*100:.1f}%")
            print(f"  - 累计收益: {total_return:.2f}%")
            print(f"  - 平均每轮收益: {total_return/len(self.period_results):.2f}%")

        # 学习到的知识
        print(f"\n学习到的知识:")
        if self.state.blacklist_hours:
            print(f"  - 黑名单时段: {self.state.blacklist_hours}")
        if self.state.blacklist_symbols:
            print(f"  - 黑名单币种: {self.state.blacklist_symbols[:10]}...")
        print(f"  - 最终阈值: {self.threshold:.2f}")

    def save(self, path: str):
        """保存系统状态"""
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_model(f"{path}/model.lgb")

        state_data = {
            'config': self.config,
            'threshold': self.threshold,
            'state': self.state.to_dict(),
            'feature_names': self.feature_engine.feature_names,
            'period_results': [asdict(r) for r in self.period_results],
        }

        with open(f"{path}/walk_forward_state.json", 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

        print(f"\n系统已保存到: {path}")


def main():
    """主程序"""
    import sys

    # 默认参数
    data_dir = '/Users/fanwei/study/quants/bian_data/uai'
    start_date = '2026-01-01'
    end_date = '2026-03-20'

    # 命令行参数
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]

    if len(sys.argv) >= 4:
        data_dir = sys.argv[3]

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       滚动前向验证系统                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  特点:                                                                        ║
║  1. 无穿越 - 永远用历史数据训练，预测未来                                       ║
║  2. 累积学习 - 训练数据持续增长，知识不断积累                                    ║
║  3. 自动进化 - 根据验证结果自动优化参数                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if not HAS_LGB:
        print("请安装lightgbm: pip install lightgbm")
        return

    system = WalkForwardSystem({
        'windows': [15, 30, 60],
        'target_profit': 3.0,
        'stop_loss': -3.0,
        'hold_minutes': 30,
        'min_precision': 0.6,
        'train_days': 10,
        'test_days': 10,
        'evolution_rounds': 3,
    })

    result = system.run_walk_forward(data_dir, start_date, end_date)

    # 保存
    save_dir = Path(data_dir) / 'backtest/walk_forward_model'
    system.save(str(save_dir))


if __name__ == '__main__':
    main()
