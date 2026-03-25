#!/usr/bin/env python3
"""
回测分析报告生成器

功能：
1. 分析回测结果
2. 识别策略不足
3. 提出改进建议
4. 生成详细报告
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import *


class BacktestAnalyzer:
    """回测结果分析器"""

    def __init__(self, results: Dict):
        self.results = results
        self.trades_df = pd.DataFrame(results.get('trades', []))
        self.daily_stats = pd.DataFrame(results.get('daily_stats', []))
        self.signals_df = pd.DataFrame(results.get('signals', []))
        self.summary = results.get('summary', {})

        self.issues = []  # 发现的问题
        self.improvements = []  # 改进建议

    def analyze_overall_performance(self) -> Dict:
        """分析整体表现"""
        analysis = {
            'performance_grade': '',
            'key_metrics': {},
            'issues': [],
        }

        win_rate = self.summary.get('win_rate', 0)
        avg_return = self.summary.get('avg_return_pct', 0)
        profit_factor = self.summary.get('profit_factor', 0)
        sharpe = self.summary.get('sharpe_ratio', 0)

        # 评级
        if win_rate >= 60 and profit_factor >= 2 and avg_return >= 1:
            analysis['performance_grade'] = 'A (优秀)'
        elif win_rate >= 50 and profit_factor >= 1.5 and avg_return >= 0.5:
            analysis['performance_grade'] = 'B (良好)'
        elif win_rate >= 40 and profit_factor >= 1 and avg_return >= 0:
            analysis['performance_grade'] = 'C (及格)'
        else:
            analysis['performance_grade'] = 'D (需改进)'

        # 问题检测
        if win_rate < 50:
            self.issues.append({
                'type': '低胜率',
                'severity': 'high',
                'detail': f'胜率仅{win_rate:.1f}%，低于50%的安全线',
                'suggestion': '提高评分阈值或增加过滤条件'
            })

        if profit_factor < 1.5:
            self.issues.append({
                'type': '盈亏比不佳',
                'severity': 'medium',
                'detail': f'盈亏比{profit_factor:.2f}，建议至少1.5以上',
                'suggestion': '调整止盈止损比例，或优化入场时机'
            })

        if avg_return < 0.5:
            self.issues.append({
                'type': '平均收益低',
                'severity': 'medium',
                'detail': f'平均每笔收益{avg_return:.2f}%，考虑手续费后可能亏损',
                'suggestion': '提高信号质量或增大止盈空间'
            })

        return analysis

    def analyze_exit_reasons(self) -> Dict:
        """分析出场原因分布"""
        if self.trades_df.empty:
            return {}

        exit_stats = self.trades_df.groupby('exit_reason').agg({
            'return_pct': ['count', 'mean', 'sum', 'std']
        }).round(2)

        analysis = {}
        for reason in exit_stats.index:
            count = exit_stats.loc[reason, ('return_pct', 'count')]
            mean_ret = exit_stats.loc[reason, ('return_pct', 'mean')]
            total_ret = exit_stats.loc[reason, ('return_pct', 'sum')]

            analysis[reason] = {
                'count': int(count),
                'percentage': count / len(self.trades_df) * 100,
                'avg_return': mean_ret,
                'total_return': total_ret
            }

        # 问题检测
        if 'timeout' in analysis:
            timeout_pct = analysis['timeout']['percentage']
            if timeout_pct > 50:
                self.issues.append({
                    'type': '超时出场过多',
                    'severity': 'medium',
                    'detail': f'{timeout_pct:.1f}%的交易以超时结束，说明行情未达预期',
                    'suggestion': '考虑缩短持仓时间或调整止盈位'
                })

        if 'stop_loss' in analysis:
            sl_pct = analysis['stop_loss']['percentage']
            if sl_pct > 30:
                self.issues.append({
                    'type': '止损过多',
                    'severity': 'high',
                    'detail': f'{sl_pct:.1f}%的交易触发止损',
                    'suggestion': '信号质量可能不足，或止损设置过紧'
                })

        return analysis

    def analyze_signal_quality(self) -> Dict:
        """分析信号质量"""
        if self.trades_df.empty:
            return {}

        # 按信号评分分组分析
        self.trades_df['score_bin'] = pd.cut(
            self.trades_df['signal_score'],
            bins=[0.9, 0.95, 0.97, 0.99, 1.0],
            labels=['0.90-0.95', '0.95-0.97', '0.97-0.99', '0.99-1.00']
        )

        score_analysis = self.trades_df.groupby('score_bin', observed=True).agg({
            'return_pct': ['count', 'mean', lambda x: (x > 0).mean() * 100]
        }).round(2)

        score_analysis.columns = ['count', 'avg_return', 'win_rate']

        # 检查是否高分信号效果更好
        analysis = score_analysis.to_dict('index')

        if len(analysis) >= 2:
            scores = list(analysis.keys())
            if analysis.get(scores[-1], {}).get('win_rate', 0) < analysis.get(scores[0], {}).get('win_rate', 0):
                self.issues.append({
                    'type': '高分信号效果不佳',
                    'severity': 'high',
                    'detail': '高评分信号的胜率反而更低，说明评分体系可能有问题',
                    'suggestion': '重新调整各因子的权重'
                })

        return analysis

    def analyze_time_patterns(self) -> Dict:
        """分析时间规律"""
        if self.trades_df.empty:
            return {}

        # 转换时间
        self.trades_df['entry_hour'] = pd.to_datetime(self.trades_df['entry_time']).dt.hour

        hour_stats = self.trades_df.groupby('entry_hour').agg({
            'return_pct': ['count', 'mean', lambda x: (x > 0).mean() * 100]
        }).round(2)
        hour_stats.columns = ['count', 'avg_return', 'win_rate']

        # 找出最佳和最差时段
        best_hour = hour_stats['avg_return'].idxmax()
        worst_hour = hour_stats['avg_return'].idxmin()

        analysis = {
            'by_hour': hour_stats.to_dict('index'),
            'best_hour': int(best_hour),
            'worst_hour': int(worst_hour),
            'best_hour_return': float(hour_stats.loc[best_hour, 'avg_return']),
            'worst_hour_return': float(hour_stats.loc[worst_hour, 'avg_return']),
        }

        # 如果某些时段表现特别差
        poor_hours = hour_stats[hour_stats['avg_return'] < -0.5].index.tolist()
        if poor_hours:
            self.issues.append({
                'type': '特定时段表现差',
                'severity': 'low',
                'detail': f'时段 {poor_hours} 平均收益为负',
                'suggestion': f'考虑在这些时段降低仓位或不交易'
            })

        return analysis

    def analyze_symbol_performance(self) -> Dict:
        """分析各币种表现"""
        if self.trades_df.empty:
            return {}

        symbol_stats = self.trades_df.groupby('symbol').agg({
            'return_pct': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
        }).round(2)
        symbol_stats.columns = ['trades', 'avg_return', 'total_return', 'win_rate']
        symbol_stats = symbol_stats.sort_values('total_return', ascending=False)

        # 找出表现最好和最差的币种
        best_symbols = symbol_stats.head(5)
        worst_symbols = symbol_stats.tail(5)

        # 检查是否有币种持续亏损
        losing_symbols = symbol_stats[
            (symbol_stats['trades'] >= 3) & (symbol_stats['win_rate'] < 30)
        ]

        if len(losing_symbols) > 0:
            self.issues.append({
                'type': '部分币种持续亏损',
                'severity': 'medium',
                'detail': f'{len(losing_symbols)}个币种交易3次以上但胜率低于30%',
                'suggestion': '这些币种可能不适合当前策略，考虑加入黑名单'
            })

        return {
            'best_symbols': best_symbols.to_dict('index'),
            'worst_symbols': worst_symbols.to_dict('index'),
            'losing_symbols': losing_symbols.index.tolist() if len(losing_symbols) > 0 else []
        }

    def analyze_drawdown(self) -> Dict:
        """分析回撤情况"""
        if self.trades_df.empty:
            return {}

        # 计算累计收益曲线
        self.trades_df['cumulative_return'] = self.trades_df['return_pct'].cumsum()

        # 计算最大回撤
        peak = self.trades_df['cumulative_return'].expanding().max()
        drawdown = self.trades_df['cumulative_return'] - peak
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()

        # 连续亏损分析
        self.trades_df['is_loss'] = self.trades_df['return_pct'] < 0
        consecutive_losses = []
        current_streak = 0
        for is_loss in self.trades_df['is_loss']:
            if is_loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_losses.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consecutive_losses.append(current_streak)

        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0

        analysis = {
            'max_drawdown_pct': float(max_drawdown),
            'max_consecutive_losses': max_consecutive_losses,
            'avg_consecutive_losses': np.mean(consecutive_losses) if consecutive_losses else 0,
        }

        if max_drawdown < -10:
            self.issues.append({
                'type': '最大回撤过大',
                'severity': 'high',
                'detail': f'最大回撤达{max_drawdown:.1f}%',
                'suggestion': '考虑降低单次仓位或增加止损'
            })

        if max_consecutive_losses >= 5:
            self.issues.append({
                'type': '连续亏损过多',
                'severity': 'medium',
                'detail': f'最大连续亏损{max_consecutive_losses}次',
                'suggestion': '考虑在连续亏损后暂停交易或降低仓位'
            })

        return analysis

    def generate_improvements(self) -> List[Dict]:
        """生成改进建议"""
        improvements = []

        # 基于问题生成改进建议
        severity_order = {'high': 1, 'medium': 2, 'low': 3}
        sorted_issues = sorted(self.issues, key=lambda x: severity_order.get(x['severity'], 4))

        for issue in sorted_issues:
            improvements.append({
                'priority': issue['severity'],
                'issue': issue['type'],
                'detail': issue['detail'],
                'action': issue['suggestion']
            })

        # 通用改进建议
        improvements.extend([
            {
                'priority': 'general',
                'issue': '参数优化',
                'detail': '当前参数基于有限数据，可能不是最优',
                'action': '使用网格搜索或遗传算法优化 score_threshold, 止盈止损比例等参数'
            },
            {
                'priority': 'general',
                'issue': '特征工程',
                'detail': '当前仅使用成交量、买卖比、净买入三个特征',
                'action': '考虑加入: 波动率、订单簿深度、资金费率、大单占比等特征'
            },
            {
                'priority': 'general',
                'issue': '市场状态识别',
                'detail': '策略在不同市场状态下表现可能不同',
                'action': '加入市场状态判断（趋势/震荡），在不同状态下使用不同参数'
            },
        ])

        self.improvements = improvements
        return improvements

    def generate_report(self) -> str:
        """生成完整分析报告"""
        # 运行所有分析
        overall = self.analyze_overall_performance()
        exit_analysis = self.analyze_exit_reasons()
        signal_analysis = self.analyze_signal_quality()
        time_analysis = self.analyze_time_patterns()
        symbol_analysis = self.analyze_symbol_performance()
        drawdown_analysis = self.analyze_drawdown()
        improvements = self.generate_improvements()

        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("                    回测分析报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 整体表现
        report.append("=" * 80)
        report.append("一、整体表现")
        report.append("=" * 80)
        report.append(f"评级: {overall['performance_grade']}")
        report.append("")
        report.append("核心指标:")
        report.append(f"  - 总交易次数: {self.summary.get('total_trades', 0)}")
        report.append(f"  - 胜率: {self.summary.get('win_rate', 0):.1f}%")
        report.append(f"  - 总收益: {self.summary.get('total_return_pct', 0):.2f}%")
        report.append(f"  - 平均收益: {self.summary.get('avg_return_pct', 0):.2f}%")
        report.append(f"  - 盈亏比: {self.summary.get('profit_factor', 0):.2f}")
        report.append(f"  - 夏普比率: {self.summary.get('sharpe_ratio', 0):.2f}")
        report.append("")

        # 出场分析
        report.append("=" * 80)
        report.append("二、出场原因分析")
        report.append("=" * 80)
        for reason, stats in exit_analysis.items():
            report.append(f"  {reason}:")
            report.append(f"    次数: {stats['count']} ({stats['percentage']:.1f}%)")
            report.append(f"    平均收益: {stats['avg_return']:.2f}%")
            report.append(f"    总收益: {stats['total_return']:.2f}%")
        report.append("")

        # 回撤分析
        report.append("=" * 80)
        report.append("三、风险分析")
        report.append("=" * 80)
        report.append(f"  - 最大回撤: {drawdown_analysis.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"  - 最大连续亏损: {drawdown_analysis.get('max_consecutive_losses', 0)}次")
        report.append("")

        # 时间分析
        if time_analysis:
            report.append("=" * 80)
            report.append("四、时间规律")
            report.append("=" * 80)
            report.append(f"  - 最佳时段: {time_analysis.get('best_hour', 'N/A')}:00 "
                         f"(平均收益 {time_analysis.get('best_hour_return', 0):.2f}%)")
            report.append(f"  - 最差时段: {time_analysis.get('worst_hour', 'N/A')}:00 "
                         f"(平均收益 {time_analysis.get('worst_hour_return', 0):.2f}%)")
            report.append("")

        # 问题汇总
        report.append("=" * 80)
        report.append("五、发现的问题")
        report.append("=" * 80)
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(issue['severity'], '⚪')
                report.append(f"  {i}. {severity_emoji} [{issue['severity'].upper()}] {issue['type']}")
                report.append(f"     详情: {issue['detail']}")
                report.append(f"     建议: {issue['suggestion']}")
                report.append("")
        else:
            report.append("  未发现明显问题")
        report.append("")

        # 改进建议
        report.append("=" * 80)
        report.append("六、改进建议")
        report.append("=" * 80)
        for i, imp in enumerate(improvements, 1):
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'general': '💡'}.get(imp['priority'], '⚪')
            report.append(f"  {i}. {priority_emoji} {imp['issue']}")
            report.append(f"     问题: {imp['detail']}")
            report.append(f"     行动: {imp['action']}")
            report.append("")

        # 表现最差的币种
        if symbol_analysis.get('losing_symbols'):
            report.append("=" * 80)
            report.append("七、建议加入黑名单的币种")
            report.append("=" * 80)
            for symbol in symbol_analysis['losing_symbols']:
                report.append(f"  - {symbol}")
            report.append("")

        report.append("=" * 80)
        report.append("报告结束")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """主程序"""
    # 查找最新的回测结果
    results_dir = Path(OUTPUT_DIR)
    result_files = list(results_dir.glob('backtest_results_*.json'))

    if not result_files:
        print("未找到回测结果文件，请先运行 backtest_engine.py")
        return

    # 使用最新的结果
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"加载回测结果: {latest_file}")

    with open(latest_file) as f:
        results = json.load(f)

    # 分析
    analyzer = BacktestAnalyzer(results)
    report = analyzer.generate_report()

    # 打印报告
    print(report)

    # 保存报告
    report_file = results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_file}")

    # 保存详细分析结果
    analysis_results = {
        'overall': analyzer.analyze_overall_performance(),
        'exit_reasons': analyzer.analyze_exit_reasons(),
        'signal_quality': analyzer.analyze_signal_quality(),
        'time_patterns': analyzer.analyze_time_patterns(),
        'symbol_performance': analyzer.analyze_symbol_performance(),
        'drawdown': analyzer.analyze_drawdown(),
        'issues': analyzer.issues,
        'improvements': analyzer.improvements,
    }

    analysis_file = results_dir / f"analysis_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
