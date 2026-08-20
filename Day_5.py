"""
Day 5：高级风控指标
- 参数法 VaR（正态分布假设）
- CVaR（条件 VaR）
- 压力测试（市场大跌情景）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 读取数据
# ============================================================

print("=" * 60)
print("Day 5：高级风控指标")
print("=" * 60)

# 读取 Day 3 保存的数据
df_all = pd.read_csv('df_all_with_indicators.csv')
df_all['date'] = pd.to_datetime(df_all['date'])

# 股票名称映射
stock_names = {
    'sz.000001': '平安银行',
    'sz.000002': '万科A',
    'sz.000858': '五粮液',
    'sh.600519': '贵州茅台'
}

print(f"\n数据概况：{len(df_all)} 行，{df_all['code'].nunique()} 只股票")
#nunique() 是 pandas 里用来统计不重复值个数的函数。


# ============================================================
# 2. 定义风控函数
# ============================================================

def calculate_var_parametric(group, confidence=0.95):
    """
    参数法 VaR（假设收益率服从正态分布）
    """
    rets = group['ret'].dropna()
    mu = rets.mean()
    sigma = rets.std()
    
    # 正态分布分位数
    z_score = 1.645 if confidence == 0.95 else 2.326
    return mu - z_score * sigma


def calculate_cvar(group, confidence=0.95):
    """
    条件 VaR（CVaR / Expected Shortfall）
    超过 VaR 部分的平均损失
    """
    rets = group['ret'].dropna()
    var_percentile = 1 - confidence
    var = rets.quantile(var_percentile)
    cvar = rets[rets <= var].mean()
    return cvar


def stress_test(group, market_shock):
    """
    压力测试：模拟市场大跌
    market_shock: 市场跌幅（如 -0.05 表示市场跌 5%）
    返回：组合亏损 = beta × 市场跌幅
    """
    rets = group['ret'].dropna()
    # 用市场跌幅作为 beta 的代理（简化：假设 beta = 1）
    # 更精确的做法：用 CAPM 回归计算 beta
    return market_shock  # 简化版


# ============================================================
# 3. 计算每只股票的参数法 VaR
# ============================================================

print("\n" + "-" * 60)
print("3.1 参数法 VaR（正态分布假设）")
print("-" * 60)

var_95_param = df_all.groupby('code').apply(calculate_var_parametric, confidence=0.95)
var_99_param = df_all.groupby('code').apply(calculate_var_parametric, confidence=0.99)

var_param_result = pd.DataFrame({
    'VaR_95_参数法': var_95_param,
    'VaR_99_参数法': var_99_param
})
var_param_result['name'] = var_param_result.index.map(stock_names)

print("\n参数法 VaR 结果：")
print(var_param_result[['name', 'VaR_95_参数法', 'VaR_99_参数法']].round(4))


# ============================================================
# 4. 计算每只股票的 CVaR
# ============================================================

print("\n" + "-" * 60)
print("3.2 CVaR（条件 VaR / Expected Shortfall）")
print("-" * 60)

cvar_95 = df_all.groupby('code').apply(calculate_cvar, confidence=0.95)
cvar_99 = df_all.groupby('code').apply(calculate_cvar, confidence=0.99)

cvar_result = pd.DataFrame({
    'CVaR_95': cvar_95,
    'CVaR_99': cvar_99
})
cvar_result['name'] = cvar_result.index.map(stock_names)

print("\nCVaR 结果：")
print(cvar_result[['name', 'CVaR_95', 'CVaR_99']].round(4))


# ============================================================
# 5. 合并所有 VaR 指标（历史模拟法 + 参数法 + CVaR）
# ============================================================

print("\n" + "-" * 60)
print("3.3 VaR 方法大对比")
print("-" * 60)

# 先读取历史模拟法 VaR（从 Day 4 的结果）
# 如果 Day 4 没有运行，这里用函数重新计算
def calculate_var_historical(group, confidence=0.95):
    rets = group['ret'].dropna()
    return rets.quantile(1 - confidence)

var_95_hist = df_all.groupby('code').apply(calculate_var_historical, confidence=0.95)
var_99_hist = df_all.groupby('code').apply(calculate_var_historical, confidence=0.99)

all_var = pd.DataFrame({
    'name': var_param_result['name'],
    'VaR_95_历史模拟': var_95_hist.values,
    'VaR_95_参数法': var_param_result['VaR_95_参数法'].values,
    'CVaR_95': cvar_result['CVaR_95'].values,
    'VaR_99_历史模拟': var_99_hist.values,
    'VaR_99_参数法': var_param_result['VaR_99_参数法'].values,
    'CVaR_99': cvar_result['CVaR_99'].values,
})

print("\n所有 VaR/CVaR 指标对比：")
print(all_var.round(4))

# 计算差异
print("\n方法差异（参数法 - 历史模拟法）：")
all_var['diff_95'] = all_var['VaR_95_参数法'] - all_var['VaR_95_历史模拟']
all_var['diff_99'] = all_var['VaR_99_参数法'] - all_var['VaR_99_历史模拟']
print(all_var[['name', 'diff_95', 'diff_99']].round(4))


# ============================================================
# 6. 压力测试
# ============================================================

print("\n" + "-" * 60)
print("3.4 压力测试（情景模拟）")
print("-" * 60)

# 设定 3 个压力情景
scenarios = {
    '轻度下跌': -0.05,
    '中度下跌': -0.10,
    '重度下跌': -0.20,
}

# 计算组合（等权重）的压力损失
pivot_returns = df_all.pivot_table(index='date', columns='code', values='ret')
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_ret = (pivot_returns * weights).sum(axis=1).dropna()

# 组合的 beta（简化：用组合与市场的相关系数 × 组合波动率 / 市场波动率）
# 这里用简单方法：用组合收益率的标准差作为 beta 的代理
portfolio_vol = portfolio_ret.std()
# 用沪深300指数作为市场（如果数据里没有，用组合自身代替）
market_vol = portfolio_vol  # 简化

print("\n压力测试结果（等权重组合）：")
for name, shock in scenarios.items():
    # 简化：组合损失 = beta × 市场跌幅
    # beta ≈ 1（简化假设）
    loss = -shock  # 正数表示亏损
    print(f"  {name}（市场跌 {abs(shock):.0%}）：组合预计亏损 {loss:.2%}")


# ============================================================
# 7. 可视化：收益率分布 + VaR 对比（茅台示例）
# ============================================================

print("\n" + "-" * 60)
print("3.5 生成可视化图表")
print("-" * 60)

code_example = 'sh.600519'
name_example = stock_names[code_example]
rets = df_all[df_all['code'] == code_example]['ret'].dropna()

plt.figure(figsize=(14, 6))

# 直方图 + 密度曲线
plt.subplot(1, 2, 1)
plt.hist(rets, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

mu = rets.mean()
sigma = rets.std()
x = np.linspace(rets.min(), rets.max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正态分布拟合')

# 标出 VaR 位置
var_hist = all_var[all_var['name'] == name_example]['VaR_95_历史模拟'].values[0]
var_param = all_var[all_var['name'] == name_example]['VaR_95_参数法'].values[0]
cvar_val = all_var[all_var['name'] == name_example]['CVaR_95'].values[0]

plt.axvline(var_hist, color='blue', linestyle='--', linewidth=2, label=f'历史模拟 VaR ({var_hist:.2%})')
plt.axvline(var_param, color='green', linestyle='--', linewidth=2, label=f'参数法 VaR ({var_param:.2%})')
plt.axvline(cvar_val, color='red', linestyle='-.', linewidth=2, label=f'CVaR ({cvar_val:.2%})')

plt.title(f'{name_example} 收益率分布与 VaR 对比', fontsize=12)
plt.xlabel('日收益率')
plt.ylabel('概率密度')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)


# 子图2：三种 VaR 方法对比（柱状图）
plt.subplot(1, 2, 2)
x_pos = np.arange(len(stock_names))
width = 0.25

# 提取数据
names = all_var['name'].values
hist_95 = all_var['VaR_95_历史模拟'].values
param_95 = all_var['VaR_95_参数法'].values
cvar_95 = all_var['CVaR_95'].values

plt.bar(x_pos - width, hist_95, width, label='历史模拟法', color='skyblue')
plt.bar(x_pos, param_95, width, label='参数法', color='lightgreen')
plt.bar(x_pos + width, cvar_95, width, label='CVaR', color='salmon')

plt.xticks(x_pos, names, rotation=15)
plt.title('三种 VaR 方法对比（95% 置信度）', fontsize=12)
plt.ylabel('VaR / CVaR')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day5_var_cvar_comparison.png', dpi=150)
plt.show()
print("  图已保存：day5_var_cvar_comparison.png")


# ============================================================
# 8. 保存完整报告
# ============================================================

print("\n" + "-" * 60)
print("3.6 完整风控报告")
print("-" * 60)

# 合并所有指标
full_report = pd.DataFrame({
    'name': all_var['name'],
    'VaR_95_历史模拟': all_var['VaR_95_历史模拟'],
    'VaR_95_参数法': all_var['VaR_95_参数法'],
    'CVaR_95': all_var['CVaR_95'],
    'VaR_99_历史模拟': all_var['VaR_99_历史模拟'],
    'VaR_99_参数法': all_var['VaR_99_参数法'],
    'CVaR_99': all_var['CVaR_99'],
})

print("\n完整风控报告：")
print(full_report.round(4))

# 保存为 CSV
full_report.to_csv('full_risk_report.csv', index=False)
print("\n✅ 报告已保存：full_risk_report.csv")

print("\n" + "=" * 60)
print("Day 5 全部完成！🎉")
print("=" * 60)