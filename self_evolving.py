#!/usr/bin/env python3
"""
自我进化回测系统

核心思路：
┌─────────────────────────────────────────────────────────────────────────────┐
│                           自我进化循环                                       │
│                                                                             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │  数据    │───▶│  训练    │───▶│  回测    │───▶│  分析    │           │
│    │  准备    │    │  模型    │    │  验证    │    │  问题    │           │
│    └──────────┘    └──────────┘    └──────────┘    └────┬─────┘           │
│         ▲                                               │                  │
│         │          ┌──────────┐    ┌──────────┐        │                  │
│         └──────────│  重新    │◀───│  自动    │◀───────┘                  │
│                    │  训练    │    │  优化    │                            │
│                    └──────────┘    └──────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘

自动优化内容：
1. 特征选择 - 剔除无用特征，保留重要特征
2. 超参数调优 - 使用Optuna自动搜索
3. 过滤规则 - 自动添加时段/币种过滤
4. 止盈止损 - 动态调整
5. 信号阈值 - 根据精确率/召回率平衡调整
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import warnings
import time
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ============================================================
# 数据结构
# ============================================================
@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    entry_time: str
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str
    signal_proba: float
    hour: int = 0

@dataclass
class EvolutionState:
    """进化状态"""
    generation: int = 0
    best_score: float = 0.0
    best_params: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    blacklist_symbols: List[str] = field(default_factory=list)
    blacklist_hours: List[int] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ============================================================
# 特征工程
# ============================================================
class FeatureEngine:
    """特征引擎"""

    def __init__(self, windows: List[int] = [15, 30, 60]):
        self.windows = windows
        self.feature_names = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        result = df.copy()

        result['net_buy'] = result['buy_volume'] - result['sell_volume']
        result['bsr'] = result['buy_volume'] / result['sell_volume'].replace(0, 0.001)
        result['volatility'] = (result['high'] - result['low']) / result['close']

        features = []

        for w in self.windows:
            # Z-score
            for col in ['volume', 'bsr', 'net_buy', 'volatility']:
                fname = f'{col}_z_{w}'
                mean = result[col].rolling(w).mean()
                std = result[col].rolling(w).std().replace(0, 1)
                result[fname] = (result[col] - mean) / std
                features.append(fname)

            # 百分位
            for col in ['volume', 'bsr', 'net_buy']:
                fname = f'{col}_pct_{w}'
                result[fname] = result[col].rolling(w+1).apply(
                    lambda x: (x[:-1] < x[-1]).sum() / (len(x)-1) if len(x) > 1 else 0.5,
                    raw=True
                )
                features.append(fname)

            # 价格位置
            fname = f'price_pos_{w}'
            high_roll = result['high'].rolling(w).max()
            low_roll = result['low'].rolling(w).min()
            result[fname] = (result['close'] - low_roll) / (high_roll - low_roll).replace(0, 1)
            features.append(fname)

            # 动量
            fname = f'momentum_{w}'
            result[fname] = result['close'] / result['close'].shift(w) - 1
            features.append(fname)

            # 买入压力
            fname = f'buy_pressure_{w}'
            result[fname] = (result['net_buy'] > 0).rolling(w).mean()
            features.append(fname)

            # 成交量比率
            fname = f'vol_ratio_{w}'
            result[fname] = result['volume'] / result['volume'].rolling(w).mean().replace(0, 1)
            features.append(fname)

        self.feature_names = features
        return result


# ============================================================
# 自我进化系统
# ============================================================
class SelfEvolvingSystem:
    """自我进化回测系统"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'windows': [15, 30, 60],
            'target_profit': 3.0,
            'stop_loss': -3.0,
            'hold_minutes': 30,
            'min_precision': 0.6,  # 最低精确率要求
            'min_trades_per_day': 1,  # 每天最少交易数
            'max_generations': 5,  # 最大进化代数
        }

        self.feature_engine = FeatureEngine(self.config['windows'])
        self.state = EvolutionState()
        self.model = None
        self.threshold = 0.5

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """加载并预处理数据"""
        all_data = []

        for fp in file_paths:
            df = pd.read_csv(fp)
            df['datetime'] = pd.to_datetime(df['transact_time'], unit='ms')
            df['minute'] = df['datetime'].dt.floor('1min')

            # 提取symbol
            symbol = Path(fp).stem.split('-')[0]

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
            agg['date'] = agg['minute'].dt.date.astype(str)

            all_data.append(agg)

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

            # 确保是同一个symbol
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

    def train_model(self, X: pd.DataFrame, y: pd.Series, params: Dict = None) -> Dict:
        """训练模型"""
        if not HAS_LGB:
            raise ImportError("需要安装lightgbm")

        params = params or {
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

            # 评分：精确率 * sqrt(交易数)，鼓励更多高质量交易
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
            'feature_importance': importance
        }

    def backtest(self, df: pd.DataFrame, features: List[str]) -> List[TradeRecord]:
        """执行回测"""
        if self.model is None:
            raise ValueError("模型未训练")

        trades = []
        X = df[features]

        # 预测
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
        symbol_cooldown = defaultdict(int)
        cooldown = self.config['hold_minutes']

        for i in np.where(signals)[0]:
            symbol = df.iloc[i]['symbol']

            # 冷却检查
            if i < symbol_cooldown[symbol]:
                continue

            if i + self.config['hold_minutes'] >= len(df):
                continue

            # 确保同一symbol
            if df.iloc[i + self.config['hold_minutes']]['symbol'] != symbol:
                continue

            entry_price = df.iloc[i]['close'] * 1.001  # 滑点

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

            ret = (exit_price - entry_price) / entry_price * 100 - 0.08  # 手续费

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

    def analyze_and_evolve(self, trades: List[TradeRecord]) -> Dict:
        """分析结果并进化"""
        if not trades:
            return {'action': 'no_trades', 'suggestions': ['降低阈值或放宽条件']}

        trades_df = pd.DataFrame([asdict(t) for t in trades])

        analysis = {
            'total_trades': len(trades),
            'win_rate': (trades_df['return_pct'] > 0).mean(),
            'total_return': trades_df['return_pct'].sum(),
            'avg_return': trades_df['return_pct'].mean(),
            'issues': [],
            'actions': [],
        }

        # 1. 分析时段表现
        hour_stats = trades_df.groupby('hour')['return_pct'].agg(['count', 'mean', 'sum'])
        bad_hours = hour_stats[hour_stats['mean'] < -0.5].index.tolist()

        if bad_hours:
            analysis['issues'].append(f"时段 {bad_hours} 表现差")
            # 自动添加到黑名单
            for h in bad_hours:
                if h not in self.state.blacklist_hours:
                    self.state.blacklist_hours.append(h)
            analysis['actions'].append(f"已将时段 {bad_hours} 加入黑名单")

        # 2. 分析币种表现
        symbol_stats = trades_df.groupby('symbol')['return_pct'].agg(['count', 'mean', 'sum'])
        bad_symbols = symbol_stats[(symbol_stats['count'] >= 3) & (symbol_stats['mean'] < -0.5)].index.tolist()

        if bad_symbols:
            analysis['issues'].append(f"币种 {bad_symbols} 表现差")
            for s in bad_symbols:
                if s not in self.state.blacklist_symbols:
                    self.state.blacklist_symbols.append(s)
            analysis['actions'].append(f"已将币种 {bad_symbols} 加入黑名单")

        # 3. 分析出场原因
        exit_stats = trades_df.groupby('exit_reason')['return_pct'].agg(['count', 'mean'])

        if 'stop_loss' in exit_stats.index:
            sl_ratio = exit_stats.loc['stop_loss', 'count'] / len(trades_df)
            if sl_ratio > 0.3:
                analysis['issues'].append(f"止损比例过高 ({sl_ratio:.1%})")
                # 调整止损
                self.config['stop_loss'] = max(self.config['stop_loss'] - 0.5, -5.0)
                analysis['actions'].append(f"放宽止损至 {self.config['stop_loss']}%")

        if 'timeout' in exit_stats.index:
            timeout_ratio = exit_stats.loc['timeout', 'count'] / len(trades_df)
            if timeout_ratio > 0.6:
                analysis['issues'].append(f"超时出场过多 ({timeout_ratio:.1%})")
                # 延长持仓或降低止盈
                self.config['target_profit'] = max(self.config['target_profit'] - 0.5, 1.5)
                analysis['actions'].append(f"降低止盈至 {self.config['target_profit']}%")

        # 4. 分析精确率
        precision = (trades_df['return_pct'] > 0).mean()
        if precision < self.config['min_precision']:
            analysis['issues'].append(f"精确率不足 ({precision:.1%})")
            # 提高阈值
            self.threshold = min(self.threshold + 0.05, 0.8)
            analysis['actions'].append(f"提高信号阈值至 {self.threshold:.2f}")

        # 5. 分析特征重要性，剔除无用特征
        if self.state.feature_importance:
            importance = self.state.feature_importance
            avg_importance = np.mean(list(importance.values()))
            weak_features = [f for f, v in importance.items() if v < avg_importance * 0.1]

            if weak_features and len(weak_features) < len(importance) * 0.3:
                analysis['issues'].append(f"发现 {len(weak_features)} 个弱特征")
                analysis['actions'].append(f"下一代将剔除: {weak_features[:5]}...")

        # 更新进化状态
        score = analysis['total_return'] * analysis['win_rate']

        if score > self.state.best_score:
            self.state.best_score = score
            self.state.best_params = {
                'threshold': self.threshold,
                'target_profit': self.config['target_profit'],
                'stop_loss': self.config['stop_loss'],
                'blacklist_hours': self.state.blacklist_hours.copy(),
                'blacklist_symbols': self.state.blacklist_symbols.copy(),
            }

        self.state.generation += 1
        self.state.history.append({
            'generation': self.state.generation,
            'score': score,
            'win_rate': analysis['win_rate'],
            'total_return': analysis['total_return'],
            'n_trades': len(trades),
            'actions': analysis['actions']
        })

        return analysis

    def evolve(self, file_paths: List[str], max_generations: int = None) -> Dict:
        """执行进化循环"""
        max_gen = max_generations or self.config['max_generations']

        print("=" * 70)
        print("自我进化系统启动")
        print("=" * 70)

        # 加载数据
        print("\n[数据加载]")
        data = self.load_data(file_paths)
        print(f"  加载 {len(file_paths)} 个文件, {len(data)} 条记录")
        print(f"  涉及 {data['symbol'].nunique()} 个币种")

        # 计算特征
        print("\n[特征计算]")
        data = self.feature_engine.compute(data)
        print(f"  生成 {len(self.feature_engine.feature_names)} 个特征")

        # 生成标签
        print("\n[标签生成]")
        data['label'] = self.generate_labels(data)
        valid_data = data.dropna(subset=['label'] + self.feature_engine.feature_names)
        print(f"  有效样本: {len(valid_data)}, 正样本率: {valid_data['label'].mean():.2%}")

        X = valid_data[self.feature_engine.feature_names]
        y = valid_data['label'].astype(int)

        best_result = None

        for gen in range(max_gen):
            print(f"\n{'='*70}")
            print(f"第 {gen + 1} 代进化")
            print(f"{'='*70}")

            # 训练
            print("\n[训练模型]")
            train_result = self.train_model(X, y)
            print(f"  阈值: {train_result['threshold']:.2f}")
            print(f"  精确率: {train_result['precision']:.2%}")
            print(f"  召回率: {train_result['recall']:.2%}")

            # 回测
            print("\n[回测验证]")
            trades = self.backtest(valid_data, self.feature_engine.feature_names)

            if trades:
                wins = sum(1 for t in trades if t.return_pct > 0)
                total_ret = sum(t.return_pct for t in trades)
                print(f"  交易数: {len(trades)}")
                print(f"  胜率: {wins/len(trades)*100:.1f}%")
                print(f"  总收益: {total_ret:.2f}%")
            else:
                print("  无交易")

            # 分析与进化
            print("\n[分析与进化]")
            analysis = self.analyze_and_evolve(trades)

            if analysis.get('issues'):
                print(f"  发现问题:")
                for issue in analysis['issues']:
                    print(f"    - {issue}")

            if analysis.get('actions'):
                print(f"  执行动作:")
                for action in analysis['actions']:
                    print(f"    - {action}")

            # 记录最佳结果
            if trades:
                score = sum(t.return_pct for t in trades) * (sum(1 for t in trades if t.return_pct > 0) / len(trades))
                if best_result is None or score > best_result['score']:
                    best_result = {
                        'generation': gen + 1,
                        'score': score,
                        'trades': len(trades),
                        'win_rate': sum(1 for t in trades if t.return_pct > 0) / len(trades),
                        'total_return': sum(t.return_pct for t in trades),
                        'params': {
                            'threshold': self.threshold,
                            'target_profit': self.config['target_profit'],
                            'stop_loss': self.config['stop_loss'],
                        }
                    }

            # 检查是否需要继续
            if not analysis.get('actions'):
                print("\n  无需进一步优化，提前结束")
                break

        # 最终结果
        print(f"\n{'='*70}")
        print("进化完成")
        print(f"{'='*70}")

        if best_result:
            print(f"\n最佳结果 (第 {best_result['generation']} 代):")
            print(f"  交易数: {best_result['trades']}")
            print(f"  胜率: {best_result['win_rate']*100:.1f}%")
            print(f"  总收益: {best_result['total_return']:.2f}%")
            print(f"  最优参数: {best_result['params']}")

        if self.state.blacklist_hours:
            print(f"\n黑名单时段: {self.state.blacklist_hours}")
        if self.state.blacklist_symbols:
            print(f"黑名单币种: {self.state.blacklist_symbols}")

        return {
            'best_result': best_result,
            'evolution_history': self.state.history,
            'final_state': self.state.to_dict()
        }

    def save(self, path: str):
        """保存系统状态"""
        Path(path).mkdir(parents=True, exist_ok=True)

        # 保存模型
        if self.model:
            self.model.save_model(f"{path}/model.lgb")

        # 保存状态
        state_data = {
            'config': self.config,
            'threshold': self.threshold,
            'state': self.state.to_dict(),
            'feature_names': self.feature_engine.feature_names,
        }

        with open(f"{path}/state.json", 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

        print(f"系统已保存到: {path}")

    def load(self, path: str):
        """加载系统状态"""
        with open(f"{path}/state.json") as f:
            state_data = json.load(f)

        self.config = state_data['config']
        self.threshold = state_data['threshold']
        self.state = EvolutionState.from_dict(state_data['state'])
        self.feature_engine.feature_names = state_data['feature_names']

        if Path(f"{path}/model.lgb").exists():
            self.model = lgb.Booster(model_file=f"{path}/model.lgb")

        print(f"系统已加载: {path}")
        print(f"  当前代数: {self.state.generation}")
        print(f"  最佳得分: {self.state.best_score:.2f}")


# ============================================================
# 主程序
# ============================================================
def main():
    """测试自我进化系统"""
    from pathlib import Path

    # 查找数据文件
    data_dir = Path('/Users/fanwei/study/quants/bian_data/uai')
    files = list(data_dir.glob('*-aggTrades-*.csv'))

    if not files:
        print("未找到数据文件")
        return

    if not HAS_LGB:
        print("请安装lightgbm: pip install lightgbm")
        return

    print(f"找到 {len(files)} 个数据文件")

    # 创建系统
    system = SelfEvolvingSystem({
        'windows': [15, 30, 60],
        'target_profit': 3.0,
        'stop_loss': -3.0,
        'hold_minutes': 30,
        'min_precision': 0.6,
        'max_generations': 3,
    })

    # 运行进化
    result = system.evolve([str(f) for f in files])

    # 保存
    system.save(str(data_dir / 'backtest/evolved_model'))

    return result


if __name__ == '__main__':
    main()
