import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import baostock as bs

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Day 3：多股票量化分析")
print("=" * 50)

# ============================================================
# 第一部分：下载多只股票数据
# ============================================================

print("\n【1】正在下载数据...")

# 登录 baostock
lg = bs.login()

# 定义股票列表（4只不同行业的股票）
stock_list = ['sz.000001', 'sz.000002', 'sz.000858', 'sh.600519']
stock_names = {
    'sz.000001': '平安银行',
    'sz.000002': '万科A',
    'sz.000858': '五粮液',
    'sh.600519': '贵州茅台'
}

all_data = []

for code in stock_list:
    print(f"  下载 {stock_names[code]} ({code})...")
    rs = bs.query_history_k_data_plus(
        code,
        "date,close",
        start_date='2020-01-01',
        end_date='2024-12-31',
        frequency="d",
        adjustflag="2"
    )
    
    # 提取数据
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    temp = pd.DataFrame(data_list, columns=rs.fields)
    
    # 添加股票代码列（重要！）
    temp['code'] = code
    
    # 添加股票名称列（方便查看）
    temp['name'] = stock_names[code]
    
    all_data.append(temp)

bs.logout()
print("  数据下载完成！")

# 合并成一张大表
df_all = pd.concat(all_data, ignore_index=True)

# 数据清洗
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['close'] = df_all['close'].astype(float)

# 按 code 和 date 排序
df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)

print(f"\n数据概况：")
print(f"  总行数：{len(df_all)}")
print(f"  股票数量：{df_all['code'].nunique()}")
print(f"  日期范围：{df_all['date'].min()} 至 {df_all['date'].max()}")
print(f"\n前5行预览：")
print(df_all.head())

# ============================================================
# 第二部分：批量计算指标（用 groupby）
# ============================================================

print("\n【2】正在计算指标...")

def calc_indicators(group):
    """
    对单只股票的数据，计算收益率、均线、波动率
    """
    group = group.sort_values('date').copy()
    
    # 收益率
    group['ret'] = group['close'].pct_change()
    group['log_ret'] = np.log(group['close'] / group['close'].shift(1))
    
    # 移动平均线
    group['ma5'] = group['close'].rolling(5).mean()
    group['ma20'] = group['close'].rolling(20).mean()
    group['ma60'] = group['close'].rolling(60).mean()
    
    # 波动率（20日，年化）
    group['vol20'] = group['ret'].rolling(20).std() * np.sqrt(252)
    
    return group

# 批量计算（关键：groupby + apply）
df_all = df_all.groupby('code').apply(calc_indicators).reset_index(drop=True)

print("  指标计算完成！")
print(df_all[['code', 'name', 'date', 'close', 'ma20', 'vol20']].head(10))

# ============================================================
# 第三部分：查看每只股票的最新指标
# ============================================================

print("\n【3】各股票最新指标：")

# 取每只股票最新一天的数据
latest = df_all.groupby('code').last()[['name', 'close', 'ma20', 'vol20']]
print(latest.round(2))

# ============================================================
# 第四部分：波动率排序
# ============================================================

print("\n【4】波动率排序：")
print("  波动率最高的股票：")
print(latest.sort_values('vol20', ascending=False)[['name', 'vol20']].round(2))

print("\n  波动率最低的股票：")
print(latest.sort_values('vol20', ascending=True)[['name', 'vol20']].round(2))

# ============================================================
# 第五部分：计算夏普比率
# ============================================================

print("\n【5】夏普比率计算：")

def calc_sharpe(group):
    """
    计算单只股票的夏普比率（假设无风险利率=0）
    """
    mean_ret = group['log_ret'].mean() * 252
    std_ret = group['log_ret'].std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret != 0 else 0
    return pd.Series({
        '年化收益': mean_ret,
        '年化波动': std_ret,
        '夏普比率': sharpe
    })

sharpe_df = df_all.groupby('code').apply(calc_sharpe)
sharpe_df['name'] = sharpe_df.index.map(stock_names)
print(sharpe_df[['name', '年化收益', '年化波动', '夏普比率']].round(4).sort_values('夏普比率', ascending=False))

# ============================================================
# 第六部分：可视化对比
# ============================================================

print("\n【6】正在生成图表...")

# 图1：净值曲线对比
plt.figure(figsize=(14, 6))

for code in stock_list:
    temp = df_all[df_all['code'] == code]
    # 归一化：以第一天为1
    normalized = temp['close'] / temp['close'].iloc[0]
    plt.plot(temp['date'], normalized, label=stock_names[code], linewidth=1.5)

plt.legend(loc='upper left')
plt.title('各股票净值走势对比（归一化）', fontsize=14)
plt.xlabel('日期')
plt.ylabel('净值（起始=1）')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_净值对比.png', dpi=150)
plt.show()
print("  图1已保存：day3_净值对比.png")

# 图2：波动率对比
plt.figure(figsize=(14, 6))

for code in stock_list:
    temp = df_all[df_all['code'] == code]
    plt.plot(temp['date'], temp['vol20'], label=stock_names[code], alpha=0.7, linewidth=1.5)

plt.legend(loc='upper left')
plt.title('各股票20日波动率对比', fontsize=14)
plt.xlabel('日期')
plt.ylabel('年化波动率')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_波动率对比.png', dpi=150)
plt.show()
print("  图2已保存：day3_波动率对比.png")

# 图3：收盘价 vs MA20
plt.figure(figsize=(14, 6))

maotai = df_all[df_all['code'] == 'sh.600519']
plt.plot(maotai['date'], maotai['close'], label='收盘价', color='black', linewidth=1.5)
plt.plot(maotai['date'], maotai['ma20'], label='MA20', color='red', linewidth=1.5)
plt.plot(maotai['date'], maotai['ma60'], label='MA60', color='blue', linewidth=1.5)

plt.legend(loc='upper left')
plt.title('贵州茅台 收盘价与均线', fontsize=14)
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_茅台均线.png', dpi=150)
plt.show()
print("  图3已保存：day3_茅台均线.png")

plt.figure(figsize=(14, 6))

wuliang = df_all[df_all['code'] == 'sz.000858']
plt.plot(wuliang['date'], wuliang['close'], label='收盘价', color='black', linewidth=1.5)
plt.plot(wuliang['date'], wuliang['ma20'], label='MA20', color='red', linewidth=1.5)
plt.plot(wuliang['date'], wuliang['ma60'], label='MA60', color='blue', linewidth=1.5)

plt.legend(loc='upper left')
plt.title('五粮液 收盘价与均线', fontsize=14)
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_五粮液均线.png', dpi=150)
plt.show()
print("  图3已保存：day3_五粮液均线.png")

plt.figure(figsize=(14, 6))

wanke = df_all[df_all['code'] == 'sz.000002']
plt.plot(wanke['date'], wanke['close'], label='收盘价', color='black', linewidth=1.5)
plt.plot(wanke['date'], wanke['ma20'], label='MA20', color='red', linewidth=1.5)
plt.plot(wanke['date'], wanke['ma60'], label='MA60', color='blue', linewidth=1.5)

plt.legend(loc='upper left')
plt.title('万科A 收盘价与均线', fontsize=14)
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_万科均线.png', dpi=150)
plt.show()
print("  图3已保存：day3_万科均线.png")

plt.figure(figsize=(14, 6))

pingan = df_all[df_all['code'] == 'sz.000001']
plt.plot(pingan['date'], pingan['close'], label='收盘价', color='black', linewidth=1.5)
plt.plot(pingan['date'], pingan['ma20'], label='MA20', color='red', linewidth=1.5)
plt.plot(pingan['date'], pingan['ma60'], label='MA60', color='blue', linewidth=1.5)

plt.legend(loc='upper left')
plt.title('平安银行 收盘价与均线', fontsize=14)
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('day3_平安均线.png', dpi=150)
plt.show()
print("  图3已保存：day3_平安均线.png")

# ============================================================
# 第七部分：保存数据
# ============================================================

print("\n【7】保存数据...")
df_all.to_csv('df_all_with_indicators.csv', index=False)
print("  数据已保存：df_all_with_indicators.csv")

# ============================================================
# 第八部分：简单筛选示例
# ============================================================

print("\n【8】简单筛选示例：")

# 筛选 2024-01-02 各股票的表现
specific_date = '2024-01-02'
day_data = df_all[df_all['date'] == specific_date]
if len(day_data) > 0:
    print(f"  {specific_date} 各股票表现：")
    print(day_data[['name', 'close', 'ma20', 'vol20']].round(2))

# 选出收盘价 > MA20 的股票（多头排列）
latest = df_all.groupby('code').last()
strong_stocks = latest[latest['close'] > latest['ma20']]
if len(strong_stocks) > 0:
    print(f"\n  多头排列（收盘价 > MA20）的股票：")
    for code in strong_stocks.index:
        print(f"    {stock_names[code]}")

print("\n" + "=" * 50)
print("Day 3 全部完成！")
print("=" * 50)

# ============================================================
# 模块一：风控核心指标
# ============================================================

print("\n" + "=" * 50)
print("风控模块：最大回撤 & VaR")
print("=" * 50)


def calculate_max_drawdown(group):
    """
    计算单只股票的最大回撤
    """
    # 归一化净值（起始=1）
    group = group.sort_values('date').copy()
    group['nav'] = (1 + group['ret']).cumprod()  # 净值曲线
    
    # 历史最高净值（累积最大值）
    group['peak'] = group['nav'].cummax()
    
    # 回撤 = (当前净值 - 历史最高) / 历史最高
    group['drawdown'] = (group['nav'] - group['peak']) / group['peak']
    
    # 最大回撤
    max_drawdown = group['drawdown'].min()
    
    # 当前回撤（最近一天）
    current_drawdown = group['drawdown'].iloc[-1]
    
    return pd.Series({
        '最大回撤': max_drawdown,
        '当前回撤': current_drawdown,
        '当前净值': group['nav'].iloc[-1]
    })


# 计算每只股票的最大回撤
drawdown_df = df_all.groupby('code').apply(calculate_max_drawdown)
drawdown_df['name'] = drawdown_df.index.map(stock_names)

print("\n各股票回撤情况：")
print(drawdown_df[['name', '最大回撤', '当前回撤', '当前净值']].round(4))

def calculate_var_historical(group, confidence=0.95, holding_period=1):
    """
    历史模拟法计算 VaR
    confidence: 置信度（0.95 或 0.99）
    holding_period: 持有期（天）
    """
    # 取收益率，去掉 NaN
    rets = group['ret'].dropna()
    
    # 按置信度取分位数
    # 95% VaR → 取 5% 分位数（左侧尾部的临界值）
    var_percentile = 1 - confidence
    var_daily = rets.quantile(var_percentile)
    
    # 如果持有期 > 1，乘以 sqrt(持有期)
    var_holding = var_daily * np.sqrt(holding_period)
    
    return var_holding


# 计算每只股票的 95% VaR（1天）
var_95_df = df_all.groupby('code').apply(calculate_var_historical, confidence=0.95, holding_period=1)
var_99_df = df_all.groupby('code').apply(calculate_var_historical, confidence=0.99, holding_period=1)

# 组装结果
var_result = pd.DataFrame({
    'VaR_95': var_95_df,
    'VaR_99': var_99_df
})
var_result['name'] = var_result.index.map(stock_names)

print("\n各股票 VaR（历史模拟法）：")
print(var_result[['name', 'VaR_95', 'VaR_99']].round(4))

# 将数据转成面板格式（日期 × 股票）
pivot_returns = df_all.pivot_table(
    index='date', 
    columns='code', 
    values='ret'
)

# 等权重组合
weights = np.array([0.25, 0.25, 0.25, 0.25])  # 4只股票各25%
portfolio_ret = (pivot_returns * weights).sum(axis=1).dropna()

# 计算组合的 VaR（历史模拟法）
def calc_var_series(returns, confidence=0.95):
    var_percentile = 1 - confidence
    return returns.quantile(var_percentile)

var_portfolio_95 = calc_var_series(portfolio_ret, 0.95)
var_portfolio_99 = calc_var_series(portfolio_ret, 0.99)

# 计算组合的最大回撤
portfolio_nav = (1 + portfolio_ret).cumprod()
portfolio_peak = portfolio_nav.cummax()
portfolio_drawdown = (portfolio_nav - portfolio_peak) / portfolio_peak
portfolio_max_dd = portfolio_drawdown.min()
portfolio_current_dd = portfolio_drawdown.iloc[-1]

print("\n" + "=" * 50)
print("等权重组合风险报告：")
print("=" * 50)
print(f"  年化波动率：{portfolio_ret.std() * np.sqrt(252):.2%}")
print(f"  最大回撤：{portfolio_max_dd:.2%}")
print(f"  当前回撤：{portfolio_current_dd:.2%}")
print(f"  VaR (95%, 1天)：{var_portfolio_95:.2%}")
print(f"  VaR (99%, 1天)：{var_portfolio_99:.2%}")

# ============================================================
# 可视化回撤
# ============================================================

print("\n正在生成回撤图...")

plt.figure(figsize=(14, 6))

for code in stock_list:
    temp = df_all[df_all['code'] == code]
    temp = temp.sort_values('date').copy()
    temp['nav'] = (1 + temp['ret']).cumprod()
    temp['peak'] = temp['nav'].cummax()
    temp['drawdown'] = (temp['nav'] - temp['peak']) / temp['peak']
    plt.plot(temp['date'], temp['drawdown'], label=stock_names[code], linewidth=1.5)

plt.legend(loc='lower left')
plt.title('各股票回撤对比', fontsize=14)
plt.xlabel('日期')
plt.ylabel('回撤幅度')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('day4_回撤对比.png', dpi=150)
plt.show()
print("  回撤图已保存：day4_回撤对比.png")

# ============================================================
# 保存风控报告
# ============================================================

# 合并所有风控指标
risk_report = pd.DataFrame({
    'name': drawdown_df['name'],
    '最大回撤': drawdown_df['最大回撤'],
    '当前回撤': drawdown_df['当前回撤'],
    'VaR_95': var_result['VaR_95'],
    'VaR_99': var_result['VaR_99'],
    '年化波动率': sharpe_df['年化波动'],
    '夏普比率': sharpe_df['夏普比率'],
})
risk_report = risk_report.reset_index(drop=True)

print("\n" + "=" * 50)
print("完整风控报告：")
print("=" * 50)
print(risk_report.round(4))
print("\n✅ 风控报告已生成！")
