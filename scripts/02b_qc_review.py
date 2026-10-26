#!/usr/bin/env python3
"""
Phase 2b: QC 汇总与审阅
汇总所有 session 的 QC 信息，生成 QC 报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import yaml

# 加载配置
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DERIVATIVES_DIR = PROJECT_ROOT / config['paths']['derivatives']
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_qc_summary():
    """加载 QC 汇总数据"""
    qc_file = DERIVATIVES_DIR / "qc_v3_conservative" / "qc_summary_v3_conservative.csv"
    if not qc_file.exists():
        raise FileNotFoundError(f"QC 汇总文件不存在: {qc_file}")
    
    df = pd.read_csv(qc_file)
    return df


def generate_qc_report(df):
    """生成 QC 报告"""
    print("\n" + "="*80)
    print("Phase 2b: QC 汇总与审阅")
    print("="*80)
    
    # 1. 总体统计
    print("\n1. 总体统计")
    print("-" * 40)
    total_sessions = len(df)
    success_sessions = len(df[df['status'] == 'success'])
    print(f"总 session 数: {total_sessions}")
    print(f"成功处理: {success_sessions} ({success_sessions/total_sessions:.1%})")
    
    # 2. 坏道统计
    print("\n2. 坏道统计")
    print("-" * 40)
    print(f"平均坏道数: {df['n_bad_channels_candidate'].mean():.1f}")
    print(f"中位数坏道数: {df['n_bad_channels_candidate'].median():.0f}")
    print(f"最大坏道数: {df['n_bad_channels_candidate'].max()}")
    print(f"坏道比例 > 10%: {len(df[df['bad_channel_fraction_candidate'] > 0.10])} sessions")
    
    # 3. Epoch rejection 统计
    print("\n3. Epoch rejection 统计")
    print("-" * 40)
    success_df = df[df['status'] == 'success']
    print(f"平均初始 epoch 数: {success_df['n_epochs_initial'].mean():.0f}")
    print(f"平均保留 epoch 数: {success_df['n_epochs_retained'].mean():.0f}")
    print(f"平均 rejection 比例: {success_df['rejection_fraction'].mean():.1%}")
    print(f"Rejection > 30%: {len(success_df[success_df['rejection_fraction'] > 0.30])} sessions")
    
    # 4. ICA 统计
    print("\n4. ICA 统计")
    print("-" * 40)
    print(f"平均 ICA 成分数: {success_df['n_ica_components'].mean():.0f}")
    # 计算剔除的成分数（ica_excluded_components 是列表）
    success_df = success_df.copy()
    success_df['n_ica_excluded_count'] = success_df['ica_excluded_components'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    print(f"平均剔除成分数: {success_df['n_ica_excluded_count'].mean():.1f}")
    print(f"剔除 0 个成分: {len(success_df[success_df['n_ica_excluded_count'] == 0])} sessions")
    
    # 5. 清洁数据时长
    print("\n5. 清洁数据时长")
    print("-" * 40)
    print(f"平均清洁时长: {success_df['clean_duration_sec'].mean():.0f} s")
    print(f"中位数清洁时长: {success_df['clean_duration_sec'].median():.0f} s")
    print(f"最短清洁时长: {success_df['clean_duration_sec'].min():.0f} s")
    
    # 6. 按条件统计
    print("\n6. 按条件统计")
    print("-" * 40)
    for condition in ['NS', 'SD']:
        cond_df = success_df[success_df['condition'] == condition]
        if len(cond_df) > 0:
            print(f"\n{condition} 条件:")
            print(f"  session 数: {len(cond_df)}")
            print(f"  平均坏道数: {cond_df['n_bad_channels_candidate'].mean():.1f}")
            print(f"  平均 rejection 比例: {cond_df['rejection_fraction'].mean():.1%}")
            print(f"  平均清洁时长: {cond_df['clean_duration_sec'].mean():.0f} s")
    
    # 7. 生成可视化
    print("\n7. 生成可视化图表")
    print("-" * 40)
    
    # 图1: 坏道分布
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(success_df['n_bad_channels_candidate'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(success_df['n_bad_channels_candidate'].median(), color='red', linestyle='--', 
               label=f"Median: {success_df['n_bad_channels_candidate'].median():.0f}")
    ax.set_xlabel('Number of Bad Channels')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bad Channels per Session')
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'qc_bad_channels_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {fig_path}")
    
    # 图2: Epoch rejection 分布
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(success_df['rejection_fraction'] * 100, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(success_df['rejection_fraction'].median() * 100, color='red', linestyle='--',
               label=f"Median: {success_df['rejection_fraction'].median():.1%}")
    ax.set_xlabel('Epoch Rejection (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Epoch Rejection per Session')
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'qc_epoch_rejection_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {fig_path}")
    
    # 图3: 清洁数据时长分布
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(success_df['clean_duration_sec'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(success_df['clean_duration_sec'].median(), color='red', linestyle='--',
               label=f"Median: {success_df['clean_duration_sec'].median():.0f} s")
    ax.axvline(config['preprocessing']['min_clean_duration_sec'], color='orange', linestyle='--',
               label=f"Minimum: {config['preprocessing']['min_clean_duration_sec']} s")
    ax.set_xlabel('Clean Duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Clean Data Duration per Session')
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'qc_clean_duration_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {fig_path}")
    
    # 图4: NS vs SD 比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 坏道数
    ax = axes[0]
    for i, condition in enumerate(['NS', 'SD']):
        cond_df = success_df[success_df['condition'] == condition]
        ax.boxplot(cond_df['n_bad_channels_candidate'], positions=[i], labels=[condition])
    ax.set_ylabel('Number of Bad Channels')
    ax.set_title('Bad Channels by Condition')
    
    # Rejection 比例
    ax = axes[1]
    for i, condition in enumerate(['NS', 'SD']):
        cond_df = success_df[success_df['condition'] == condition]
        ax.boxplot(cond_df['rejection_fraction'] * 100, positions=[i], labels=[condition])
    ax.set_ylabel('Epoch Rejection (%)')
    ax.set_title('Epoch Rejection by Condition')
    
    # 清洁时长
    ax = axes[2]
    for i, condition in enumerate(['NS', 'SD']):
        cond_df = success_df[success_df['condition'] == condition]
        ax.boxplot(cond_df['clean_duration_sec'], positions=[i], labels=[condition])
    ax.set_ylabel('Clean Duration (s)')
    ax.set_title('Clean Duration by Condition')
    
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'qc_condition_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {fig_path}")
    
    # 8. 标记需要人工审阅的 session
    print("\n8. 需要人工审阅的 session")
    print("-" * 40)
    
    needs_review = []
    
    # 坏道过多
    high_bad = success_df[success_df['bad_channel_fraction_candidate'] > 0.08]
    if len(high_bad) > 0:
        print(f"\n坏道比例 > 8% ({len(high_bad)} sessions):")
        for _, row in high_bad.iterrows():
            print(f"  {row['subject']} ses-{row['session']}: "
                  f"{row['n_bad_channels_candidate']} bad channels ({row['bad_channel_fraction_candidate']:.1%})")
            needs_review.append(f"{row['subject']}_ses-{row['session']}")
    
    # Rejection 过高
    high_reject = success_df[success_df['rejection_fraction'] > 0.25]
    if len(high_reject) > 0:
        print(f"\nEpoch rejection > 25% ({len(high_reject)} sessions):")
        for _, row in high_reject.iterrows():
            print(f"  {row['subject']} ses-{row['session']}: "
                  f"{row['rejection_fraction']:.1%} rejected")
            needs_review.append(f"{row['subject']}_ses-{row['session']}")
    
    # ICA 剔除 0 个成分
    no_ica = success_df[success_df['n_ica_excluded_count'] == 0]
    if len(no_ica) > 0:
        print(f"\nICA 剔除 0 个成分 ({len(no_ica)} sessions):")
        for _, row in no_ica.iterrows():
            print(f"  {row['subject']} ses-{row['session']}")
        needs_review.extend([f"{row['subject']}_ses-{row['session']}" for _, row in no_ica.iterrows()])
    
    # 保存需要审阅的列表
    if needs_review:
        review_file = RESULTS_DIR / 'qc_needs_review.txt'
        with open(review_file, 'w') as f:
            f.write('\n'.join(set(needs_review)))
        print(f"\n需要审阅的 session 列表已保存: {review_file}")
    
    print("\n" + "="*80)
    print("QC 汇总完成")
    print("="*80)


def main():
    """主函数"""
    try:
        df = load_qc_summary()
        generate_qc_report(df)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 02a_preprocess_eeg.py")


if __name__ == '__main__':
    main()
