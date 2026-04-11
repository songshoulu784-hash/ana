"""


"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np



# ========== 1. 设置中文字体 ==========
# 关键修复：设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

mingm=pd.read_excel("C:/Users/lenovo/Desktop/游戏分析/明末评论.xls")

#删除评论页面为空白的数据
mingm_cleaned = mingm.dropna(subset=['评论页面'])

print(f"评论总条数：{mingm_cleaned.shape[0]}")
print(f"挖掘字段：{mingm_cleaned.shape[1]}")
#用户平均游戏时长（两周内）
average_playtime=mingm_cleaned["两周内时长"].mean()
print(f"用户两周内平均游戏时长：{average_playtime}")

#用户使用超过两周的人数
if '总时长'and '两周内时长' in mingm_cleaned.columns:
    users_over_twoWeeks=mingm_cleaned[mingm_cleaned["总时长"]-mingm_cleaned["两周内时长"]>0].shape[0]
    print(f"用户使用超过两周人数：{users_over_twoWeeks}")
#推荐率与不推荐率
evaluation_counts=mingm_cleaned["评价"].value_counts()
print(f"评价分布：{evaluation_counts}")

#推荐率
total_reviews = len(mingm_cleaned)
recommend_rate = evaluation_counts['推荐'] / total_reviews * 100

print(f"推荐率：{recommend_rate:.2f}%")

#已退款与未退款
print(f"退款情况分布：{mingm_cleaned['状态'].value_counts()}")

#退款率
total_rate=mingm_cleaned[mingm_cleaned['状态'] == '已退款'].shape[0]/total_reviews*100

print(f"退款率：{total_rate:.2f}%")

#退款与评价的交叉情况
def cluster_by_money(data):
    print("===退款与推荐情况 ===")

    # 退款数值 × 评价数值 的交叉表
    # table = pd.crosstab(data['退款数值'], data['评价数值'])
    # table.index = ['未退款(0)', '已退款(1)']
    # table.columns = ['不推荐(0)', '推荐(1)']
    #
    # print(table)

    summary = data.groupby(['状态', '评价']).size().reset_index(name='人数')
    print(summary)

    return summary

cluster_by_money(mingm_cleaned)

# 可视化
# 绘图（现在可以正常显示中文）
# plt.figure(figsize=(10, 6))
# plt.hist(mingm_cleaned['两周内时长'].dropna(), bins=30, edgecolor='black')
# plt.title('用户两周内游戏时长分布')
# plt.xlabel('游戏时长（小时）')
# plt.ylabel('出现次数')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('游戏时长分布.png', dpi=300, bbox_inches='tight')  # 无警告！
# plt.show()





#留存量：用户两周后还在玩同时为推荐的数量（用库存数作为分析基点）

def cluster_by_playtime(data, playtime_col='两周内时长'):
    """
    基于游戏时长进行用户分群
    """
    # 定义分群标准
    data['游戏时长分群'] = pd.cut(
        data[playtime_col],
        bins=[0, 5, 20, 50, 100, float('inf')],
        labels=['新手(0-5h)', '轻度玩家(5-20h)', '中度玩家(20-50h)',
                '重度玩家(50-100h)', '核心玩家(100h+)'],
        right=False
    )

    # 分群统计
    cluster_stats = data.groupby('游戏时长分群').agg({
        playtime_col: ['count', 'mean', 'median', 'std'],
        '评价数值': 'mean'  # 平均推荐率
    }).round(2)

    cluster_stats.columns = ['用户数', '平均时长', '时长中位数', '时长标准差', '推荐率']

    print("=== 基于游戏时长的用户分群 ===")
    print(cluster_stats)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 用户分布
    ax1 = axes[0]
    cluster_counts = data['游戏时长分群'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

    ax1.bar(cluster_counts.index, cluster_counts.values, color=colors)
    ax1.set_title('各分群用户数量分布')
    ax1.set_xlabel('用户分群')
    ax1.set_ylabel('用户数量')
    ax1.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for i, v in enumerate(cluster_counts.values):
        ax1.text(i, v + max(cluster_counts.values) * 0.01, str(v),
                 ha='center', va='bottom')


# cluster_by_playtime(mingm_cleaned)
#用户画像：高级玩家，普通玩家之类的
#推荐玩家游戏时长分布
#不推荐玩家游戏时长分布

#推荐玩家等级分布
#不推荐玩家等级分布


#游戏库存程度划分资深玩家与普通玩家

#等级划分

#评测数

#库存数