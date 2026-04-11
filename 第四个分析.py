import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 导入进阶分析库 ==========
try:
    import jieba
    import jieba.analyse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from snownlp import SnowNLP
    import gensim
    from gensim import corpora, models

    ADVANCED_AVAILABLE = True
    print("✓ 进阶分析库加载成功")
except ImportError as e:
    ADVANCED_AVAILABLE = False
    print(f"✗ 部分进阶库未安装: {e}")
    print("  请运行: pip install jieba scikit-learn snownlp gensim")

# ========== 2. 读取数据 ==========
print("\n正在读取数据...")
mingm = pd.read_excel("C:/Users/lenovo/Desktop/游戏分析/明末评论.xls")
df = mingm.dropna(subset=['评论页面']).copy()

# 填充缺失值
df['等级'] = df['等级'].fillna(0)
df['库存数'] = df['库存数'].fillna(0)
df['评测数'] = df['评测数'].fillna(0)
df['好友数'] = df['好友数'].fillna(0)
df['徽章数'] = df['徽章数'].fillna(0)
df['组数'] = df['组数'].fillna(0)
df['总时长'] = df['总时长'].fillna(0)
df['有价值数'] = df['有价值数'].fillna(0)
df['欢乐数'] = df['欢乐数'].fillna(0)
df['评论'] = df['评论'].fillna('')

print(f"✓ 数据加载完成，共 {len(df)} 条评论")

# ========== 3. 基础时间分析 ==========
release_date = datetime(2025, 7, 24)
df['发布时间'] = pd.to_datetime(df['发布时间'])
df['days_gap'] = (df['发布时间'] - release_date).dt.total_seconds() / 86400


# 时长分档（基础版）
def duration_bin(hours):
    if hours < 2:
        return '<2h (尝鲜)'
    elif hours < 5:
        return '2-5h (初步)'
    elif hours < 10:
        return '5-10h (中等)'
    elif hours < 20:
        return '10-20h (深入)'
    else:
        return '≥20h (通关+)'


df['duration_bin'] = df['总时长'].apply(duration_bin)


# 时长分档（细化版）
def duration_bin_detailed(hours):
    if hours < 1:
        return '<1h'
    elif hours < 2:
        return '1-2h'
    elif hours < 5:
        return '2-5h'
    elif hours < 10:
        return '5-10h'
    elif hours < 20:
        return '10-20h'
    elif hours < 30:
        return '20-30h'
    else:
        return '≥30h'


df['duration_bin_detailed'] = df['总时长'].apply(duration_bin_detailed)


# 时间间隔分档
def time_bin_detailed(days):
    if days < 1:
        return 'D0 (首日)'
    elif days < 2:
        return 'D1 (第2天)'
    elif days < 3:
        return 'D2 (第3天)'
    elif days < 5:
        return 'D3-4'
    elif days < 7:
        return 'D5-6'
    elif days < 10:
        return 'D7-9'
    else:
        return 'D10+'


df['time_bin'] = df['days_gap'].apply(time_bin_detailed)

# ========== 4. 评论影响力分析 ==========
df['总互动数'] = df['有价值数'] + df['欢乐数']
df['log_互动'] = np.log1p(df['总互动数'])


def influence_level(interaction):
    if interaction == 0:
        return '无互动'
    elif interaction == 1:
        return '低互动'
    elif interaction <= 3:
        return '中互动'
    else:
        return '高互动'


df['影响力等级'] = df['总互动数'].apply(influence_level)
df['高影响力'] = df['总互动数'] >= 3


# ========== 5. 核心玩家识别 ==========
def score_by_rank(value, thresholds):
    if pd.isna(value) or value == 0:
        return 0
    if value >= thresholds[3]:
        return 100
    elif value >= thresholds[2]:
        return 75
    elif value >= thresholds[1]:
        return 50
    elif value >= thresholds[0]:
        return 25
    else:
        return 0


level_thresholds = [10, 30, 60, 100]
inventory_thresholds = [50, 150, 300, 500]
review_thresholds = [10, 30, 80, 150]
friend_thresholds = [20, 50, 100, 200]
badge_thresholds = [10, 20, 40, 70]
group_thresholds = [10, 30, 60, 100]

df['等级分'] = df['等级'].apply(lambda x: score_by_rank(x, level_thresholds))
df['库存分'] = df['库存数'].apply(lambda x: score_by_rank(x, inventory_thresholds))
df['评测分'] = df['评测数'].apply(lambda x: score_by_rank(x, review_thresholds))
df['好友分'] = df['好友数'].apply(lambda x: score_by_rank(x, friend_thresholds))
df['徽章分'] = df['徽章数'].apply(lambda x: score_by_rank(x, badge_thresholds))
df['组分'] = df['组数'].apply(lambda x: score_by_rank(x, group_thresholds))

weights = {'等级分': 0.25, '库存分': 0.20, '评测分': 0.20, '好友分': 0.15, '徽章分': 0.10, '组分': 0.10}
df['核心玩家得分'] = (df['等级分'] * weights['等级分'] + df['库存分'] * weights['库存分'] +
                      df['评测分'] * weights['评测分'] + df['好友分'] * weights['好友分'] +
                      df['徽章分'] * weights['徽章分'] + df['组分'] * weights['组分'])


def player_type(score):
    if score < 20:
        return '轻度玩家'
    elif score < 50:
        return '中度玩家'
    elif score < 75:
        return '核心玩家'
    else:
        return '硬核玩家'


df['玩家类型'] = df['核心玩家得分'].apply(player_type)

# ========== 6. 评论主题分析（基础版） ==========
topic_keywords = {
    '宣发/营销': ['宣发', 'UP主', 'up主', '硬吹', '吹捧', '骗', '广告', '推广'],
    '优化/性能': ['优化', '帧', '卡顿', 'low帧', '掉帧', '配置', '显卡', '不流畅'],
    '历史/政治': ['历史', '满清', '鞑子', '扬州十日', '嘉定三屠', '屠蜀', '立场', '历史观'],
    '美术/人设': ['服饰', '卖肉', '老乡鸡', '割裂', '人设', '服装'],
    '战斗/玩法': ['技能树', 'boss', 'Boss', '红岚', '新娘', '轮椅', '交互', '血瓶', '魂类'],
    '地图/关卡': ['地图', '篝火', '电梯', '火点', 'boss房', '关卡'],
    '剧情/叙事': ['剧情', '叙事', '碎片化', '云'],
    '性价比': ['价格', '值', '248', '打折', 'CDK'],
    '退款': ['退款', 'CDK', '退不了'],
    '正面评价': ['精品', '不错', '好', '精妙', '契合']
}
positive_words = ['不错', '精妙', '精品', '好', '契合', '推荐']
negative_words = ['差', '烂', '糟糕', '脑瘫', '拙劣', '无语', '红温', '畜生']


def extract_topics(comment):
    if pd.isna(comment) or comment == '':
        return []
    topics = []
    comment_lower = str(comment).lower()
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw.lower() in comment_lower:
                topics.append(topic)
                break
    return topics


def sentiment_score_simple(comment):
    if pd.isna(comment) or comment == '':
        return 0
    comment_lower = str(comment).lower()
    pos_count = sum(1 for w in positive_words if w in comment_lower)
    neg_count = sum(1 for w in negative_words if w in comment_lower)
    return pos_count - neg_count


def comment_length_bin(comment):
    if pd.isna(comment) or comment == '':
        return '无评论'
    length = len(str(comment))
    if length < 50:
        return '短评(<50字)'
    elif length < 200:
        return '中评(50-200字)'
    else:
        return '长评(>200字)'


df['主题标签'] = df['评论'].apply(extract_topics)
df['情感得分_简单'] = df['评论'].apply(sentiment_score_simple)
df['评论长度档位'] = df['评论'].apply(comment_length_bin)
df['主题数量'] = df['主题标签'].apply(len)

# ========== 7. 进阶分析：jieba + TF-IDF ==========
if ADVANCED_AVAILABLE:
    print("\n" + "=" * 60)
    print("【进阶分析 1/4: jieba分词 + TF-IDF关键词提取】")
    print("=" * 60)

    # 停用词表
    stopwords = set([
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '也', '被', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己',
        '这', '那', '什么', '这个', '那个', '然后', '就是', '但是', '所以', '而且', '因为',
        '如果', '虽然', '但是', '还是', '或者', '不过', '只是', '真的', '非常', '比较',
        '有点', '一些', '一点', '感觉', '觉得', '可能', '可以', '应该', '需要', '想要'
    ])


    def chinese_word_seg(text):
        if pd.isna(text) or text == '':
            return []
        for char in '，。！？；：“”‘’、|【】《》~！@#￥%……&*（）——+-=：；，./<>?':
            text = text.replace(char, ' ')
        words = jieba.cut(text.strip())
        return [w for w in words if w not in stopwords and len(w) > 1]


    df['seg_words'] = df['评论'].apply(chinese_word_seg)
    df['seg_text'] = df['seg_words'].apply(lambda x: ' '.join(x))

    # 全局TF-IDF
    all_comments = df[df['评论'] != '']['评论'].tolist()
    if len(all_comments) > 0:
        tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: chinese_word_seg(x), max_features=50)
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_comments)
        global_keywords = tfidf_vectorizer.get_feature_names_out()
        global_scores = tfidf_matrix.sum(axis=0).tolist()[0]
        global_keywords_sorted = sorted(zip(global_keywords, global_scores), key=lambda x: x[1], reverse=True)[:20]

        print("\n全局TF-IDF关键词 Top 20:")
        for i, (word, score) in enumerate(global_keywords_sorted, 1):
            print(f"  {i:2d}. {word}: {score:.4f}")

        # 按评价分组的关键词
        print("\n按评价分组的关键词对比:")
        for rating in ['推荐', '不推荐']:
            rating_comments = df[df['评价'] == rating]['评论'].tolist()
            if rating_comments:
                vectorizer = TfidfVectorizer(tokenizer=lambda x: chinese_word_seg(x), max_features=15)
                tfidf_mat = vectorizer.fit_transform(rating_comments)
                keywords = vectorizer.get_feature_names_out()
                scores = tfidf_mat.sum(axis=0).tolist()[0]
                top_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:10]
                print(f"\n  【{rating}】Top 10关键词:")
                for word, score in top_keywords:
                    print(f"    - {word}: {score:.3f}")

# ========== 8. 进阶分析：SnowNLP情感分析 ==========
if ADVANCED_AVAILABLE:
    print("\n" + "=" * 60)
    print("【进阶分析 2/4: SnowNLP情感评分】")
    print("=" * 60)


    def snow_sentiment(text):
        if pd.isna(text) or text == '':
            return 0.5
        try:
            return SnowNLP(text).sentiments
        except:
            return 0.5


    df['snow_score'] = df['评论'].apply(snow_sentiment)
    df['snow_sentiment'] = df['snow_score'].apply(
        lambda x: '负面' if x < 0.4 else ('中性' if x < 0.6 else '正面')
    )

    # 对比分析
    print("\nSnowNLP情感分析 vs 实际评价对比:")
    print(f"{'实际评价':<8} {'正面':<6} {'中性':<6} {'负面':<6} {'平均分':<8} {'样本数':<6}")
    for rating in ['推荐', '不推荐']:
        subset = df[df['评价'] == rating]
        if len(subset) > 0:
            pos = (subset['snow_sentiment'] == '正面').sum()
            neu = (subset['snow_sentiment'] == '中性').sum()
            neg = (subset['snow_sentiment'] == '负面').sum()
            avg = subset['snow_score'].mean()
            print(f"{rating:<8} {pos:<6} {neu:<6} {neg:<6} {avg:.3f}     {len(subset)}")

    # 情感一致性
    df['情感_评价一致'] = ((df['评价'] == '推荐') & (df['snow_score'] > 0.5)) | \
                          ((df['评价'] == '不推荐') & (df['snow_score'] < 0.5))
    print(f"\n情感与评价一致性: {df['情感_评价一致'].mean() * 100:.1f}%")

    # 按时长分析情感
    duration_groups = ['<2h', '2-5h', '5-10h', '10-20h', '20-30h', '≥30h']
    sentiment_by_duration = df.groupby('duration_bin_detailed')['snow_score'].mean()
    print("\n按游戏时长的情感变化:")
    for duration in duration_groups:
        if duration in sentiment_by_duration.index:
            print(f"  {duration}: {sentiment_by_duration[duration]:.3f}")

# ========== 9. 进阶分析：LDA主题建模 ==========
if ADVANCED_AVAILABLE:
    print("\n" + "=" * 60)
    print("【进阶分析 3/4: LDA主题建模】")
    print("=" * 60)

    documents = df[df['评论'] != '']['seg_text'].tolist() if 'seg_text' in df.columns else []
    if len(documents) >= 10:
        texts = [doc.split() for doc in documents]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in texts]

        num_topics = min(5, max(2, len(documents) // 10))
        if num_topics >= 2:
            lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary,
                                               num_topics=num_topics, random_state=42,
                                               passes=10, iterations=100)

            print(f"\n发现 {num_topics} 个主题:")
            for idx, topic in lda_model.print_topics(num_words=8):
                print(f"\n主题 {idx + 1}:")
                words = topic.split(' + ')
                for word in words[:6]:
                    if '*' in word:
                        prob, term = word.split('*')
                        print(f"    {term.strip(chr(34))}: {float(prob):.3f}")


            # 分配主题
            def get_dominant_topic(text):
                if text == '':
                    return -1
                bow = dictionary.doc2bow(text.split())
                topics = lda_model.get_document_topics(bow)
                return max(topics, key=lambda x: x[1])[0] if topics else -1


            df['lda_topic'] = df['seg_text'].apply(get_dominant_topic) if 'seg_text' in df.columns else -1

            print("\n各主题的情感倾向:")
            for topic_id in range(num_topics):
                topic_df = df[df['lda_topic'] == topic_id]
                if len(topic_df) > 0 and 'snow_score' in df.columns:
                    print(f"  主题 {topic_id + 1}: 情感分={topic_df['snow_score'].mean():.3f}, "
                          f"好评率={(topic_df['评价'] == '推荐').mean() * 100:.1f}%, 评论数={len(topic_df)}")
    else:
        print(f"数据量不足（{len(documents)}条），跳过LDA建模")

# ========== 10. 进阶分析：BERT情感分析 ==========
print("\n" + "=" * 60)
print("【进阶分析 4/4: BERT情感分析】")
print("=" * 60)

try:
    from transformers import pipeline

    print("正在加载BERT模型...")
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model="nlptown/bert-base-multilingual-uncased-sentiment",
                                  device=-1)

    sample_comments = df[df['评论'] != '']['评论'].head(min(100, len(df))).tolist()
    if len(sample_comments) > 0:
        bert_results = []
        for i in range(0, len(sample_comments), 8):
            batch = sample_comments[i:i + 8]
            try:
                batch_results = sentiment_pipeline(batch)
                bert_results.extend(batch_results)
            except:
                bert_results.extend([{'label': '3 stars', 'score': 0.5}] * len(batch))

        star_to_score = {'1 star': 0.1, '2 stars': 0.3, '3 stars': 0.5, '4 stars': 0.7, '5 stars': 0.9}
        bert_scores = [star_to_score.get(r['label'], 0.5) for r in bert_results]

        print(f"\nBERT分析结果（{len(sample_comments)}条样本）:")
        print(f"  平均情感分: {np.mean(bert_scores):.3f}")
        print(f"  正面(>0.6): {(np.array(bert_scores) > 0.6).sum()}条")
        print(f"  负面(<0.4): {(np.array(bert_scores) < 0.4).sum()}条")

        if 'snow_score' in df.columns:
            df_sample = df.head(len(sample_comments)).copy()
            df_sample['bert_score'] = bert_scores + [0.5] * (len(df_sample) - len(bert_scores))
            correlation = df_sample['snow_score'].corr(df_sample['bert_score'])
            print(f"\nBERT与SnowNLP相关性: {correlation:.3f}")
except ImportError:
    print("需要安装: pip install transformers torch")
except Exception as e:
    print(f"BERT分析跳过: {e}")

# ========== 11. 综合可视化 ==========
print("\n" + "=" * 60)
print("【生成综合可视化图表】")
print("=" * 60)

# 图1：整体评论数量分布
fig1, ax1 = plt.subplots(figsize=(12, 5))
daily_counts = df.groupby(df['days_gap'].round(0)).size()
days = np.arange(0, max(15, int(df['days_gap'].max()) + 1))
counts = [daily_counts.get(day, 0) for day in days]
ax1.bar(days, counts, width=0.8, color='steelblue', edgecolor='black')
ax1.set_xlabel('发布后天数', fontsize=12)
ax1.set_ylabel('评论数量', fontsize=12)
ax1.set_title('《明末：渊虚之羽》Steam评论数量随时间分布', fontsize=14, fontweight='bold')
ax1.set_xticks(days[::max(1, len(days) // 10)])
ax1.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('01_整体评论数量分布.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("✓ 01_整体评论数量分布.png")

# 图2：按时长分档堆叠图
fig2, ax2 = plt.subplots(figsize=(14, 6))
pivot_stack = df.groupby([df['days_gap'].round(0), 'duration_bin']).size().unstack(fill_value=0)
pivot_stack = pivot_stack.reindex(days[:15], fill_value=0)
pivot_stack.plot(kind='bar', stacked=True, ax=ax2, colormap='Set2', edgecolor='black')
ax2.set_xlabel('发布后天数', fontsize=12)
ax2.set_ylabel('评论数量', fontsize=12)
ax2.set_title('每日评论数量（按游戏时长分档堆叠）', fontsize=14, fontweight='bold')
ax2.set_xticklabels([f'D+{d}' for d in days[:15]], rotation=45)
ax2.legend(title='游戏时长', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('02_时长分档堆叠图.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("✓ 02_时长分档堆叠图.png")

# 图3：散点图
fig3, ax3 = plt.subplots(figsize=(10, 6))
colors_rating = {'推荐': 'green', '不推荐': 'red'}
for rating, group in df.groupby('评价'):
    ax3.scatter(group['days_gap'], group['总时长'], label=rating,
                c=colors_rating.get(rating, 'gray'), s=80, alpha=0.7)
ax3.set_xlabel('发布后天数', fontsize=12)
ax3.set_ylabel('游戏总时长（小时）', fontsize=12)
ax3.set_title('游戏时长 vs 评论时间', fontsize=14, fontweight='bold')
ax3.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='2h退款线')
ax3.legend()
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('03_时长vs评论时间散点图.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("✓ 03_时长vs评论时间散点图.png")

# 图4：基础热力图
fig4, ax4 = plt.subplots(figsize=(12, 6))
time_order = ['D0 (首日)', 'D1 (第2天)', 'D2 (第3天)', 'D3-4', 'D5-6', 'D7-9', 'D10+']
duration_order = ['<1h', '1-2h', '2-5h', '5-10h', '10-20h', '20-30h', '≥30h']
pivot_table = pd.crosstab(df['time_bin'], df['duration_bin_detailed'])
pivot_table = pivot_table.fillna(0).astype(int)
pivot_table = pivot_table.reindex(index=time_order, columns=duration_order, fill_value=0)
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
            linewidths=0.5, linecolor='gray', cbar_kws={'label': '评论数量'})
ax4.set_title('评论热力图：发布时间间隔 vs 游戏时长', fontsize=14, fontweight='bold')
ax4.set_xlabel('游戏时长', fontsize=12)
ax4.set_ylabel('评论时间', fontsize=12)
plt.tight_layout()
plt.savefig('04_基础热力图.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("✓ 04_基础热力图.png")

# 图5：好评率热力图
fig5, ax5 = plt.subplots(figsize=(12, 6))
rating_matrix = pd.DataFrame(np.nan, index=time_order, columns=duration_order)
for t in time_order:
    for d in duration_order:
        subset = df[(df['time_bin'] == t) & (df['duration_bin_detailed'] == d)]
        if len(subset) > 0:
            rating_matrix.loc[t, d] = (subset['评价'] == '推荐').sum() / len(subset) * 100
sns.heatmap(rating_matrix, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax5,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=100,
            cbar_kws={'label': '好评率 (%)'})
ax5.set_title('各维度好评率热力图', fontsize=14, fontweight='bold')
ax5.set_xlabel('游戏时长', fontsize=12)
ax5.set_ylabel('评论时间', fontsize=12)
plt.tight_layout()
plt.savefig('05_好评率热力图.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print("✓ 05_好评率热力图.png")

# 图6：玩家类型分布
fig6, ax6 = plt.subplots(figsize=(10, 6))
player_counts = df['玩家类型'].value_counts().reindex(['轻度玩家', '中度玩家', '核心玩家', '硬核玩家'], fill_value=0)
colors_player = ['#90be6d', '#f9c74f', '#f9844a', '#f94144']
bars = ax6.bar(player_counts.index, player_counts.values, color=colors_player, edgecolor='black')
ax6.set_xlabel('玩家类型', fontsize=12)
ax6.set_ylabel('人数', fontsize=12)
ax6.set_title('评论者玩家类型分布', fontsize=14, fontweight='bold')
for bar, count in zip(bars, player_counts.values):
    if count > 0:
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(count)), ha='center')
ax6.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('06_玩家类型分布.png', dpi=150, bbox_inches='tight')
plt.close(fig6)
print("✓ 06_玩家类型分布.png")

# 图7：不同玩家类型的好评率
fig7, ax7 = plt.subplots(figsize=(10, 6))
pos_rate_by_type = df.groupby('玩家类型').apply(lambda x: (x['评价'] == '推荐').sum() / len(x) * 100)
pos_rate_by_type = pos_rate_by_type.reindex(['轻度玩家', '中度玩家', '核心玩家', '硬核玩家'], fill_value=0)
bars = ax7.bar(pos_rate_by_type.index, pos_rate_by_type.values, color=colors_player, edgecolor='black')
ax7.set_xlabel('玩家类型', fontsize=12)
ax7.set_ylabel('好评率 (%)', fontsize=12)
ax7.set_title('不同玩家类型的好评率', fontsize=14, fontweight='bold')
ax7.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax7.set_ylim(0, 100)
for bar, rate in zip(bars, pos_rate_by_type.values):
    ax7.text(bar.get_x() + bar.get_width() / 2, rate + 2, f'{rate:.1f}%', ha='center')
ax7.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('07_玩家类型好评率.png', dpi=150, bbox_inches='tight')
plt.close(fig7)
print("✓ 07_玩家类型好评率.png")

# 图8：SnowNLP情感分布（如有）
if ADVANCED_AVAILABLE and 'snow_score' in df.columns:
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    ax8.hist(df['snow_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax8.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='中性分界线')
    ax8.set_xlabel('情感分数 (0=负面, 1=正面)', fontsize=12)
    ax8.set_ylabel('评论数量', fontsize=12)
    ax8.set_title('SnowNLP情感分数分布', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('08_SnowNLP情感分布.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print("✓ 08_SnowNLP情感分布.png")

    # 图9：情感 vs 时长
    fig9, ax9 = plt.subplots(figsize=(12, 6))
    duration_groups = ['<2h', '2-5h', '5-10h', '10-20h', '20-30h', '≥30h']
    sentiment_means = df.groupby('duration_bin_detailed')['snow_score'].mean()
    sentiment_means = sentiment_means.reindex(duration_groups)
    ax9.bar(range(len(sentiment_means)), sentiment_means.values, color='teal', edgecolor='black')
    ax9.set_xticks(range(len(sentiment_means)))
    ax9.set_xticklabels(sentiment_means.index)
    ax9.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='中性')
    ax9.set_xlabel('游戏时长', fontsize=12)
    ax9.set_ylabel('平均情感分数', fontsize=12)
    ax9.set_title('不同游戏时长段的情感变化趋势', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('09_时长情感变化.png', dpi=150, bbox_inches='tight')
    plt.close(fig9)
    print("✓ 09_时长情感变化.png")

    # 图10：推荐vs不推荐情感箱线图
    fig10, ax10 = plt.subplots(figsize=(8, 6))
    data_to_plot = [df[df['评价'] == '推荐']['snow_score'].dropna(),
                    df[df['评价'] == '不推荐']['snow_score'].dropna()]
    bp = ax10.boxplot(data_to_plot, labels=['推荐', '不推荐'], patch_artist=True)
    for box, color in zip(bp['boxes'], ['lightgreen', 'salmon']):
        box.set_facecolor(color)
    ax10.set_ylabel('情感分数', fontsize=12)
    ax10.set_title('推荐 vs 不推荐 的情感分数对比', fontsize=14, fontweight='bold')
    ax10.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax10.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_评价情感对比.png', dpi=150, bbox_inches='tight')
    plt.close(fig10)
    print("✓ 10_评价情感对比.png")

# 图11：主题词频分布
fig11, ax11 = plt.subplots(figsize=(10, 6))
all_topics = []
for topics in df['主题标签']:
    all_topics.extend(topics)
topic_counts = Counter(all_topics)
top_topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:8])
ax11.barh(list(top_topics.keys()), list(top_topics.values()), color='coral', edgecolor='black')
ax11.set_title('评论主题词频分布', fontsize=14, fontweight='bold')
ax11.set_xlabel('出现次数', fontsize=12)
ax11.invert_yaxis()
plt.tight_layout()
plt.savefig('11_主题词频分布.png', dpi=150, bbox_inches='tight')
plt.close(fig11)
print("✓ 11_主题词频分布.png")

# 图12：推荐vs不推荐主题对比
fig12, ax12 = plt.subplots(figsize=(12, 6))
rating_topic_matrix = {}
for rating in ['推荐', '不推荐']:
    topics_in_rating = []
    for topics in df[df['评价'] == rating]['主题标签']:
        topics_in_rating.extend(topics)
    rating_topic_matrix[rating] = Counter(topics_in_rating)
topics_list = list(topic_counts.keys())
x = np.arange(len(topics_list))
width = 0.35
recommend_counts = [rating_topic_matrix['推荐'].get(t, 0) for t in topics_list]
not_recommend_counts = [rating_topic_matrix['不推荐'].get(t, 0) for t in topics_list]
ax12.bar(x - width / 2, recommend_counts, width, label='推荐', color='green', alpha=0.7)
ax12.bar(x + width / 2, not_recommend_counts, width, label='不推荐', color='red', alpha=0.7)
ax12.set_xticks(x)
ax12.set_xticklabels(topics_list, rotation=45, ha='right')
ax12.set_title('推荐 vs 不推荐 的主题分布对比', fontsize=14, fontweight='bold')
ax12.set_ylabel('出现次数', fontsize=12)
ax12.legend()
plt.tight_layout()
plt.savefig('12_推荐vs不推荐主题对比.png', dpi=150, bbox_inches='tight')
plt.close(fig12)
print("✓ 12_推荐vs不推荐主题对比.png")

# 图13：评论长度与影响力
fig13, ax13 = plt.subplots(figsize=(10, 6))
length_influence = df.groupby('评论长度档位')['总互动数'].mean()
length_influence = length_influence.reindex(['短评(<50字)', '中评(50-200字)', '长评(>200字)'], fill_value=0)
ax13.bar(length_influence.index, length_influence.values, color='steelblue', edgecolor='black')
ax13.set_title('评论长度 vs 平均影响力', fontsize=14, fontweight='bold')
ax13.set_xlabel('评论长度档位', fontsize=12)
ax13.set_ylabel('平均互动数', fontsize=12)
ax13.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('13_评论长度与影响力.png', dpi=150, bbox_inches='tight')
plt.close(fig13)
print("✓ 13_评论长度与影响力.png")

# 图14：不同影响力等级的主题数
fig14, ax14 = plt.subplots(figsize=(10, 6))
influence_order = ['无互动', '低互动', '中互动', '高互动']
avg_topics = df.groupby('影响力等级')['主题数量'].mean().reindex(influence_order, fill_value=0)
ax14.bar(avg_topics.index, avg_topics.values, color='teal', edgecolor='black')
ax14.set_title('不同影响力等级的平均主题数', fontsize=14, fontweight='bold')
ax14.set_xlabel('影响力等级', fontsize=12)
ax14.set_ylabel('平均主题数量', fontsize=12)
ax14.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('14_影响力与主题数.png', dpi=150, bbox_inches='tight')
plt.close(fig14)
print("✓ 14_影响力与主题数.png")

# ========== 12. 最终统计摘要 ==========
print("\n" + "=" * 80)
print("【完整分析报告总结】")
print("=" * 80)

print(f"\n📊 基础统计:")
print(f"  总评论数: {len(df)}")
print(f"  推荐: {len(df[df['评价'] == '推荐'])} ({len(df[df['评价'] == '推荐']) / len(df) * 100:.1f}%)")
print(f"  不推荐: {len(df[df['评价'] == '不推荐'])} ({len(df[df['评价'] == '不推荐']) / len(df) * 100:.1f}%)")
print(f"  已退款: {len(df[df['状态'] == '已退款'])}")

print(f"\n⏰ 时间分布:")
time_dist = df['days_gap'].apply(
    lambda x: '首日' if x < 1 else '第2-3天' if x < 3 else '首周' if x < 7 else '第二周' if x < 14 else '两周以上')
print(time_dist.value_counts().to_string())

print(f"\n👥 玩家类型分布:")
print(df['玩家类型'].value_counts().to_string())

print(f"\n💬 评论主题:")
print(f"  有主题标签的评论: {(df['主题数量'] > 0).sum()}/{len(df)} ({(df['主题数量'] > 0).mean() * 100:.1f}%)")
print(f"  平均每评论主题数: {df['主题数量'].mean():.2f}")
if len(topic_counts) > 0:
    print(f"  最常讨论的主题: {topic_counts.most_common(3)}")

if ADVANCED_AVAILABLE and 'snow_score' in df.columns:
    print(f"\n😊 情感分析 (SnowNLP):")
    print(f"  整体平均情感分数: {df['snow_score'].mean():.3f}")
    print(f"  正面评论比例: {(df['snow_score'] > 0.6).mean() * 100:.1f}%")
    print(f"  负面评论比例: {(df['snow_score'] < 0.4).mean() * 100:.1f}%")
    print(f"  情感与评价一致性: {df['情感_评价一致'].mean() * 100:.1f}%")

print(f"\n⭐ 各玩家类型好评率:")
for ptype in ['轻度玩家', '中度玩家', '核心玩家', '硬核玩家']:
    subset = df[df['玩家类型'] == ptype]
    if len(subset) > 0:
        pos_rate = (subset['评价'] == '推荐').sum() / len(subset) * 100
        print(f"  {ptype}: {pos_rate:.1f}% (n={len(subset)})")

print(f"\n📈 各时长档位平均互动数:")
for duration in ['<2h', '2-5h', '5-10h', '10-20h', '≥20h']:
    subset = df[df['duration_bin'] == duration]
    if len(subset) > 0:
        print(f"  {duration}: {subset['总互动数'].mean():.2f} (n={len(subset)})")

print("\n" + "=" * 80)
print("✅ 全部分析完成！")
print(f"   共生成 {len([f for f in os.listdir('.') if f.endswith('.png') and f[0:2].isdigit()])} 张图表")
print("=" * 80)

# ========== 评论修改检测与统计分析（适配版） ==========
print("\n" + "=" * 80)
print("【评论修改检测与统计分析】")
print("=" * 80)

# 1. 检查修改时间字段
if '修改时间' in df.columns:
    # 转换修改时间
    df['修改时间'] = pd.to_datetime(df['修改时间'], errors='coerce')
    df['是否有修改'] = df['修改时间'].notna()

    # 2. 计算修改延迟（发布到修改的天数）
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['修改延迟_天'] = (df['修改时间'] - df['发布时间']).dt.total_seconds() / 86400


    # 3. 标记修改类型
    def get_modification_type(row):
        if pd.isna(row['修改时间']):
            return '未修改'
        days = row['修改延迟_天']
        if days < 1:
            return '当日修改'
        elif days < 3:
            return '3日内修改'
        elif days < 7:
            return '1周内修改'
        else:
            return '1周后修改'


    df['修改类型'] = df.apply(get_modification_type, axis=1)

    # 4. 统计修改情况
    print("\n📝 修改统计概览:")
    print(f"  总评论数: {len(df)}")
    print(f"  已修改评论: {df['是否有修改'].sum()}条 ({df['是否有修改'].mean() * 100:.1f}%)")
    print(f"  未修改评论: {(~df['是否有修改']).sum()}条")

    if df['是否有修改'].sum() > 0:
        # 按评价分组的修改率
        print("\n📊 按评价分组的修改率:")
        for rating in df['评价'].unique():
            subset = df[df['评价'] == rating]
            if len(subset) > 0:
                modified_rate = subset['是否有修改'].mean() * 100
                print(f"  {rating}: {modified_rate:.1f}% ({subset['是否有修改'].sum()}/{len(subset)})")

        # 按玩家类型分组的修改率（如果已有玩家类型字段）
        if '玩家类型' in df.columns:
            print("\n👥 按玩家类型分组的修改率:")
            for ptype in df['玩家类型'].unique():
                subset = df[df['玩家类型'] == ptype]
                if len(subset) > 0 and subset['是否有修改'].sum() > 0:
                    modified_rate = subset['是否有修改'].mean() * 100
                    print(f"  {ptype}: {modified_rate:.1f}% ({subset['是否有修改'].sum()}/{len(subset)})")

        # 修改延迟分布
        print("\n⏰ 修改延迟分布:")
        modified_df = df[df['是否有修改'] == True]
        print(f"  平均修改延迟: {modified_df['修改延迟_天'].mean():.2f}天")
        print(f"  中位数修改延迟: {modified_df['修改延迟_天'].median():.2f}天")
        print(f"  最短修改延迟: {modified_df['修改延迟_天'].min():.2f}天")
        print(f"  最长修改延迟: {modified_df['修改延迟_天'].max():.2f}天")

        print("\n📋 修改类型分布:")
        print(modified_df['修改类型'].value_counts().to_string())

        # 5. 分析修改者的行为特征
        print("\n📈 修改者的行为特征分析:")

        # 比较游戏时长
        modified_avg_duration = modified_df['总时长'].mean()
        unmodified_avg_duration = df[~df['是否有修改']]['总时长'].mean()
        print(f"  修改者平均游戏时长: {modified_avg_duration:.1f}h")
        print(f"  未修改者平均游戏时长: {unmodified_avg_duration:.1f}h")

        # 比较等级
        if '等级' in df.columns:
            modified_avg_level = modified_df['等级'].mean()
            unmodified_avg_level = df[~df['是否有修改']]['等级'].mean()
            print(f"  修改者平均等级: {modified_avg_level:.1f}")
            print(f"  未修改者平均等级: {unmodified_avg_level:.1f}")

        # 比较互动数
        if '有价值数' in df.columns and '欢乐数' in df.columns:
            df['总互动数'] = df['有价值数'] + df['欢乐数']
            modified_avg_interaction = modified_df['总互动数'].mean()
            unmodified_avg_interaction = df[~df['是否有修改']]['总互动数'].mean()
            print(f"  修改者平均互动数: {modified_avg_interaction:.2f}")
            print(f"  未修改者平均互动数: {unmodified_avg_interaction:.2f}")

        # 6. 分析修改时间与游戏时长的关系
        print("\n🎮 修改时间与游戏时长的关系:")
        if 'duration_bin' in df.columns:
            duration_modify_rate = df.groupby('duration_bin')['是否有修改'].mean() * 100
            for duration, rate in duration_modify_rate.items():
                if rate > 0:
                    print(f"  {duration}: {rate:.1f}%")
        else:
            # 动态分档
            def simple_duration_bin(hours):
                if hours < 2:
                    return '<2h'
                elif hours < 5:
                    return '2-5h'
                elif hours < 10:
                    return '5-10h'
                elif hours < 20:
                    return '10-20h'
                else:
                    return '≥20h'


            df['temp_duration_bin'] = df['总时长'].apply(simple_duration_bin)
            duration_modify_rate = df.groupby('temp_duration_bin')['是否有修改'].mean() * 100
            for duration, rate in duration_modify_rate.items():
                if rate > 0:
                    print(f"  {duration}: {rate:.1f}%")

        # 7. 分析修改时间与评论时间的关系
        print("\n📅 修改时间与评论发布时间的关系:")
        df['comment_week'] = df['days_gap'].apply(lambda x: f'第{int(x // 7) + 1}周' if x < 21 else '3周以上')
        weekly_modify_rate = df.groupby('comment_week')['是否有修改'].mean() * 100
        for week, rate in weekly_modify_rate.items():
            if rate > 0:
                print(f"  {week}: {rate:.1f}%")

else:
    print("\n⚠️ 数据中没有'修改时间'字段，无法进行修改检测")
    print("   请确保数据包含修改时间列")

# ========== 修改检测可视化 ==========
print("\n" + "=" * 80)
print("【生成修改检测可视化图表】")
print("=" * 80)

if '修改时间' in df.columns and df['是否有修改'].sum() > 0:

    # 图15：修改率饼图
    fig15, ax15 = plt.subplots(figsize=(8, 8))
    modified_counts = [df['是否有修改'].sum(), (~df['是否有修改']).sum()]
    labels = [f'已修改\n({modified_counts[0]}条)', f'未修改\n({modified_counts[1]}条)']
    colors_pie = ['#ff9999', '#66b3ff']
    ax15.pie(modified_counts, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90,
             explode=(0.05, 0), shadow=True)
    ax15.set_title('评论修改率分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('15_评论修改率分布.png', dpi=150, bbox_inches='tight')
    plt.close(fig15)
    print("✓ 15_评论修改率分布.png")

    # 图16：修改延迟分布直方图
    fig16, ax16 = plt.subplots(figsize=(12, 6))
    modified_df = df[df['是否有修改'] == True]
    ax16.hist(modified_df['修改延迟_天'], bins=20, color='teal', edgecolor='black', alpha=0.7)
    ax16.axvline(x=modified_df['修改延迟_天'].mean(), color='red', linestyle='--',
                 linewidth=2, label=f"平均: {modified_df['修改延迟_天'].mean():.1f}天")
    ax16.axvline(x=modified_df['修改延迟_天'].median(), color='orange', linestyle='--',
                 linewidth=2, label=f"中位数: {modified_df['修改延迟_天'].median():.1f}天")
    ax16.set_xlabel('修改延迟（天）', fontsize=12)
    ax16.set_ylabel('评论数量', fontsize=12)
    ax16.set_title('评论修改延迟分布', fontsize=14, fontweight='bold')
    ax16.legend()
    ax16.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('16_修改延迟分布.png', dpi=150, bbox_inches='tight')
    plt.close(fig16)
    print("✓ 16_修改延迟分布.png")

    # 图17：不同评价类型的修改率对比
    fig17, ax17 = plt.subplots(figsize=(8, 6))
    rating_modify_rate = df.groupby('评价')['是否有修改'].mean() * 100
    colors_bar = ['green' if r == '推荐' else 'red' for r in rating_modify_rate.index]
    bars = ax17.bar(rating_modify_rate.index, rating_modify_rate.values, color=colors_bar, edgecolor='black')
    ax17.set_ylabel('修改率 (%)', fontsize=12)
    ax17.set_xlabel('评价类型', fontsize=12)
    ax17.set_title('不同评价类型的修改率对比', fontsize=14, fontweight='bold')
    ax17.set_ylim(0, max(rating_modify_rate.values, default=0) * 1.2)
    for bar, rate in zip(bars, rating_modify_rate.values):
        ax17.text(bar.get_x() + bar.get_width() / 2, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
    ax17.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('17_评价类型修改率对比.png', dpi=150, bbox_inches='tight')
    plt.close(fig17)
    print("✓ 17_评价类型修改率对比.png")

    # 图18：不同玩家类型的修改率（如果有）
    if '玩家类型' in df.columns:
        fig18, ax18 = plt.subplots(figsize=(10, 6))
        player_modify_rate = df.groupby('玩家类型')['是否有修改'].mean() * 100
        player_modify_rate = player_modify_rate.reindex(['轻度玩家', '中度玩家', '核心玩家', '硬核玩家'], fill_value=0)
        colors_player = ['#90be6d', '#f9c74f', '#f9844a', '#f94144']
        bars = ax18.bar(player_modify_rate.index, player_modify_rate.values, color=colors_player, edgecolor='black')
        ax18.set_ylabel('修改率 (%)', fontsize=12)
        ax18.set_xlabel('玩家类型', fontsize=12)
        ax18.set_title('不同玩家类型的修改率对比', fontsize=14, fontweight='bold')
        ax18.set_ylim(0, max(player_modify_rate.values, default=0) * 1.2)
        for bar, rate in zip(bars, player_modify_rate.values):
            if rate > 0:
                ax18.text(bar.get_x() + bar.get_width() / 2, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
        ax18.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('18_玩家类型修改率对比.png', dpi=150, bbox_inches='tight')
        plt.close(fig18)
        print("✓ 18_玩家类型修改率对比.png")

    # 图19：不同时长档位的修改率
    fig19, ax19 = plt.subplots(figsize=(12, 6))
    if 'duration_bin' in df.columns:
        duration_modify_rate = df.groupby('duration_bin')['是否有修改'].mean() * 100
    else:
        df['temp_duration_bin'] = df['总时长'].apply(simple_duration_bin)
        duration_modify_rate = df.groupby('temp_duration_bin')['是否有修改'].mean() * 100

    bars = ax19.bar(range(len(duration_modify_rate)), duration_modify_rate.values, color='steelblue', edgecolor='black')
    ax19.set_xticks(range(len(duration_modify_rate)))
    ax19.set_xticklabels(duration_modify_rate.index, rotation=45)
    ax19.set_ylabel('修改率 (%)', fontsize=12)
    ax19.set_xlabel('游戏时长档位', fontsize=12)
    ax19.set_title('不同游戏时长档位的修改率', fontsize=14, fontweight='bold')
    ax19.set_ylim(0, max(duration_modify_rate.values, default=0) * 1.2)
    for bar, rate in zip(bars, duration_modify_rate.values):
        if rate > 0:
            ax19.text(bar.get_x() + bar.get_width() / 2, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
    ax19.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('19_时长档位修改率.png', dpi=150, bbox_inches='tight')
    plt.close(fig19)
    print("✓ 19_时长档位修改率.png")

    # 图20：修改时间趋势
    fig20, ax20 = plt.subplots(figsize=(12, 6))
    modified_by_date = modified_df.groupby(modified_df['修改时间'].dt.date).size()
    if len(modified_by_date) > 0:
        ax20.plot(range(len(modified_by_date)), modified_by_date.values, marker='o', color='purple', linewidth=2,
                  markersize=8)
        ax20.set_xticks(range(len(modified_by_date)))
        ax20.set_xticklabels([str(d)[5:10] for d in modified_by_date.index], rotation=45)
        ax20.set_xlabel('修改日期', fontsize=12)
        ax20.set_ylabel('修改数量', fontsize=12)
        ax20.set_title('评论修改数量随时间变化趋势', fontsize=14, fontweight='bold')
        ax20.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('20_修改时间趋势.png', dpi=150, bbox_inches='tight')
        plt.close(fig20)
        print("✓ 20_修改时间趋势.png")

# ========== 详细修改记录输出 ==========
if '修改时间' in df.columns and df['是否有修改'].sum() > 0:
    print("\n" + "=" * 80)
    print("【详细修改记录】")
    print("=" * 80)

    modified_list = df[df['是否有修改'] == True].sort_values('修改延迟_天')
    print(f"\n共 {len(modified_list)} 条修改记录:\n")

    for idx, row in modified_list.iterrows():
        print(f"【{row['用户名']}】")
        print(f"  评价: {row['评价']}")
        if '玩家类型' in df.columns:
            print(f"  玩家类型: {row['玩家类型']}")
        print(f"  发布时间: {row['发布时间'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['发布时间']) else '未知'}")
        print(f"  修改时间: {row['修改时间'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['修改时间']) else '未知'}")
        print(f"  修改延迟: {row['修改延迟_天']:.1f}天")
        print(f"  游戏时长: {row['总时长']}h")
        if '有价值数' in df.columns and '欢乐数' in df.columns:
            print(f"  互动数: {row['有价值数'] + row['欢乐数']}")
        print(f"  评论预览: {str(row['评论'])[:80]}..." if len(str(row['评论'])) > 80 else f"  评论: {row['评论']}")
        print("-" * 50)

# ========== 修改分析总结 ==========
print("\n" + "=" * 80)
print("【修改检测分析总结】")
print("=" * 80)

if '修改时间' in df.columns:
    if df['是否有修改'].sum() > 0:
        print("\n🔍 关键发现:")

        # 找出修改率最高的群体
        if '玩家类型' in df.columns:
            highest_modify_group = df.groupby('玩家类型')['是否有修改'].mean().idxmax()
            highest_modify_rate = df.groupby('玩家类型')['是否有修改'].mean().max() * 100
            print(f"  1. {highest_modify_group}的修改率最高 ({highest_modify_rate:.1f}%)")

        # 找出修改最快的评论
        fastest_modify = modified_df.loc[modified_df['修改延迟_天'].idxmin()]
        print(f"  2. 最快修改: {fastest_modify['用户名']} 在 {fastest_modify['修改延迟_天']:.1f}天后修改")

        # 修改与互动的关系
        if '总互动数' in df.columns:
            if modified_df['总互动数'].mean() > df[~df['是否有修改']]['总互动数'].mean():
                print(
                    f"  3. 修改过的评论平均互动数更高 ({modified_df['总互动数'].mean():.2f} vs {df[~df['是否有修改']]['总互动数'].mean():.2f})")
            else:
                print(
                    f"  3. 修改过的评论平均互动数较低 ({modified_df['总互动数'].mean():.2f} vs {df[~df['是否有修改']]['总互动数'].mean():.2f})")

        # 时长与修改的关系
        if modified_df['总时长'].mean() > df[~df['是否有修改']]['总时长'].mean():
            print(
                f"  4. 修改者平均游戏时长更长 ({modified_df['总时长'].mean():.1f}h vs {df[~df['是否有修改']]['总时长'].mean():.1f}h)")
        else:
            print(
                f"  4. 修改者平均游戏时长更短 ({modified_df['总时长'].mean():.1f}h vs {df[~df['是否有修改']]['总时长'].mean():.1f}h)")

        # 推荐与不推荐的修改率差异
        if '评价' in df.columns:
            rec_modify = df[df['评价'] == '推荐']['是否有修改'].mean() * 100 if len(df[df['评价'] == '推荐']) > 0 else 0
            not_rec_modify = df[df['评价'] == '不推荐']['是否有修改'].mean() * 100 if len(
                df[df['评价'] == '不推荐']) > 0 else 0
            print(f"  5. 不推荐评论修改率: {not_rec_modify:.1f}% | 推荐评论修改率: {rec_modify:.1f}%")

        # 额外洞察
        print(f"\n💡 额外洞察:")
        print(f"  - 修改延迟中位数: {modified_df['修改延迟_天'].median():.1f}天")
        print(
            f"  - 大部分修改发生在: {modified_df['修改类型'].mode().values[0] if len(modified_df['修改类型'].mode()) > 0 else '未知'}")

    else:
        print("\n📌 当前数据中没有检测到被修改的评论")
        print("   （注：需要数据中包含'修改时间'字段且有实际修改记录）")
else:
    print("\n📌 数据中没有'修改时间'字段，无法进行修改检测")
    print("   如需此功能，请确保数据包含修改时间列")

print("\n" + "=" * 80)
print("✅ 修改检测分析完成！")
if '修改时间' in df.columns and df['是否有修改'].sum() > 0:
    print(
        "   新增图表: 15_评论修改率分布.png, 16_修改延迟分布.png, 17_评价类型修改率对比.png, 18_玩家类型修改率对比.png, 19_时长档位修改率.png, 20_修改时间趋势.png")
print("=" * 80)