#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# FACTOR ZOO: MULTI-FACTOR RETURN ATTRIBUTION
# Replicates Fama-French 5-Factor + Momentum Model
# ═══════════════════════════════════════════════════════════════════

# ── CELL 1 — INSTALL & IMPORTS ──────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import yfinance as yf
import warnings
import datetime

warnings.filterwarnings('ignore')

# Color palette — consistent throughout all charts
C = {
    'blue':   '#1f77b4',
    'red':    '#d62728',
    'green':  '#2ca02c',
    'orange': '#ff7f0e',
    'purple': '#9467bd',
    'grey':   '#7f7f7f',
    'teal':   '#17becf',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        11,
})

print(f"✅ Setup complete | {datetime.date.today()}")

# ── CELL 2 — CONFIGURATION ──────────────────────────────────────────

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'JNJ',
           'XOM',  'AMZN', 'PG',   'UNH', 'META']

WEIGHTS = [0.15, 0.15, 0.10, 0.10, 0.10,
           0.10, 0.10, 0.10, 0.05, 0.05]

BENCHMARK  = 'SPY'
START_DATE = '2015-01-01'
END_DATE   = datetime.date.today().strftime('%Y-%m-%d')

# Sanity checks
assert len(TICKERS) == len(WEIGHTS), "Tickers and weights must have equal length"
assert abs(sum(WEIGHTS) - 1.0) < 1e-9, f"Weights sum to {sum(WEIGHTS):.4f}, must be 1.0"

print("Portfolio:")
for t, w in zip(TICKERS, WEIGHTS):
    print(f"  {t:<6} {w*100:.0f}%")
print(f"\nBenchmark : {BENCHMARK}")
print(f"Period    : {START_DATE} → {END_DATE}")

# ── CELL 3 — FETCH FAMA-FRENCH FACTORS ──────────────────────────────

def fetch_ff_factors(start, end):
    ff5 = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench',
        start=start, end=end
    )[0]

    mom = web.DataReader(
        'F-F_Momentum_Factor', 'famafrench',
        start=start, end=end
    )[0]
    mom.columns = ['Mom']

    factors = ff5.join(mom, how='inner') / 100
    factors.index = factors.index.to_timestamp()
    return factors

print("Fetching Fama-French factors...")
factors = fetch_ff_factors(START_DATE, END_DATE)

print(f"✅ {len(factors)} months loaded | {factors.index[0].date()} to {factors.index[-1].date()}")
print(f"\nColumns: {list(factors.columns)}")
print(f"\nAnnualized factor means:")
for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']:
    ann = factors[col].mean() * 12
    print(f"  {col:<8} {ann*100:>+6.2f}% / yr")

# ── CELL 4 — FETCH STOCK PRICES & BUILD PORTFOLIO RETURNS ───────────

def fetch_monthly_returns(tickers, benchmark, start, end):
    all_tickers = tickers + [benchmark]

    raw = yf.download(all_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']] if len(all_tickers) == 1 else raw

    try:
        monthly = prices.resample('ME').last()
    except ValueError:
        monthly = prices.resample('M').last()

    monthly.index = monthly.index.to_period('M').to_timestamp()

    returns = monthly.pct_change().dropna()
    return returns

print("Downloading prices from Yahoo Finance...")
all_returns = fetch_monthly_returns(TICKERS, BENCHMARK, START_DATE, END_DATE)

stock_returns = all_returns[TICKERS]
bench_returns = all_returns[BENCHMARK]

print(f"✅ {len(all_returns)} months | {stock_returns.shape[1]} stocks + benchmark")

missing = stock_returns.isnull().sum()
if missing.any():
    print(f"\n⚠️  Missing data:\n{missing[missing > 0]}")
else:
    print("✅ No missing values")

# ── CELL 5 — BUILD PORTFOLIO RETURN SERIES & ALIGN WITH FACTORS ─────

def build_portfolio_returns(stock_rets, weights):
    w = np.array(weights)
    port_ret = stock_rets.values @ w
    return pd.Series(port_ret, index=stock_rets.index, name='Portfolio')

portfolio_returns = build_portfolio_returns(stock_returns, WEIGHTS)

common_idx = (factors.index
              .intersection(portfolio_returns.index)
              .intersection(bench_returns.index))

F = factors.loc[common_idx]
P = portfolio_returns.loc[common_idx]
B = bench_returns.loc[common_idx]

print(f"Aligned dataset: {common_idx[0].date()} → {common_idx[-1].date()} ({len(common_idx)} months)\n")

def ann_stats(ret_series, rf_series):
    excess  = ret_series - rf_series
    cum     = (1 + ret_series).prod() - 1
    ann_ret = (1 + cum) ** (12 / len(ret_series)) - 1 if len(ret_series) > 0 else 0
    ann_vol = ret_series.std() * np.sqrt(12)
    sharpe  = excess.mean() / excess.std() * np.sqrt(12)
    roll_max = (1 + ret_series).cumprod().cummax()
    max_dd  = ((1 + ret_series).cumprod() / roll_max - 1).min()
    return ann_ret, ann_vol, sharpe, cum, max_dd

p_ret, p_vol, p_sr, p_cum, p_dd = ann_stats(P, F['RF'])
b_ret, b_vol, b_sr, b_cum, b_dd = ann_stats(B, F['RF'])

print(f"{'Metric':<22} {'Portfolio':>12} {'SPY':>10}")
print("-" * 46)
print(f"{'Ann. Return':<22} {p_ret*100:>11.2f}%  {b_ret*100:>9.2f}%")
print(f"{'Ann. Volatility':<22} {p_vol*100:>11.2f}%  {b_vol*100:>9.2f}%")
print(f"{'Sharpe Ratio':<22} {p_sr:>12.2f}  {b_sr:>9.2f}")
print(f"{'Cumulative Return':<22} {p_cum*100:>11.2f}%  {b_cum*100:>9.2f}%")
print(f"{'Max Drawdown':<22} {p_dd*100:>11.2f}%  {b_dd*100:>9.2f}%")

# ── CELL 6 — FACTOR REGRESSION: CORE MODEL ──────────────────────────

FACTOR_SETS = {
    'CAPM': ['Mkt-RF'],
    'FF3' : ['Mkt-RF', 'SMB', 'HML'],
    'FF6' : ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'],
}

def run_ols(excess_returns, factors_df, factor_cols):
    X = sm.add_constant(factors_df[factor_cols])
    return sm.OLS(excess_returns, X).fit()

excess_P = P - F['RF']

models = {
    name: run_ols(excess_P, F, cols)
    for name, cols in FACTOR_SETS.items()
}

print(f"\n{'Model':<8} {'Ann. Alpha':>12} {'t(α)':>8} {'R²':>8} {'Adj-R²':>9}")
print("─" * 50)
for name, m in models.items():
    alpha_ann = m.params['const'] * 12
    t_alpha   = m.tvalues['const']
    print(f"{name:<8} {alpha_ann*100:>10.2f}%   {t_alpha:>+6.2f}   {m.rsquared:.3f}   {m.rsquared_adj:.3f}")

# ── CELL 7 — DETAILED FF6 RESULTS TABLE ─────────────────────────────

def print_regression_table(model, model_name):
    params = model.params
    tvals  = model.tvalues
    pvals  = model.pvalues
    ci     = model.conf_int()

    print(f"\n{'='*65}")
    print(f"  {model_name}  |  R² = {model.rsquared:.3f}  |  n = {int(model.nobs)} months")
    print(f"{'='*65}")
    print(f"{'Factor':<20} {'Loading':>10} {'t-stat':>8} {'p-val':>8} {'95% CI':>20} {'Sig':>4}")
    print("─" * 65)

    for var in params.index:
        stars = ('***' if pvals[var] < 0.01 else
                 '**'  if pvals[var] < 0.05 else
                 '*'   if pvals[var] < 0.10 else '')

        label = 'Alpha (monthly)' if var == 'const' else var
        val   = params[var]
        lo, hi = ci.loc[var, 0], ci.loc[var, 1]
        print(f"{label:<20} {val:>+10.4f} {tvals[var]:>+8.2f} {pvals[var]:>8.3f} [{lo:>+6.3f}, {hi:>+6.3f}]  {stars:>4}")

    alpha_bps = model.params['const'] * 10000 * 12
    sig_flag  = "✅ significant" if abs(model.tvalues['const']) > 2 else "⚠️  not significant"
    print(f"\n  Annualized alpha: {alpha_bps:+.0f} bps/yr  ({sig_flag}, t = {model.tvalues['const']:+.2f})")
    print(f"  Interpretation: {model.rsquared*100:.1f}% of portfolio variance explained by factors.")

print_regression_table(models['FF6'], "FF6 Model: 5-Factor + Momentum")

# ── CELL 8 — RETURN ATTRIBUTION DECOMPOSITION ───────────────────────

def decompose_attribution(model, factors_df, factor_cols, total_months):
    params       = model.params
    factor_means = factors_df[factor_cols].mean()

    contribs = {'Risk-Free Rate': factors_df['RF'].mean() * total_months,
                'Alpha': params['const'] * total_months}
    for f in factor_cols:
        contribs[f] = params[f] * factor_means[f] * total_months

    return contribs

T = len(F)
contribs = decompose_attribution(models['FF6'], F, FACTOR_SETS['FF6'], T)

total_ret   = P.sum()
model_total = sum(contribs.values())
residual    = total_ret - model_total
contribs['Residual'] = residual

print(f"Return Attribution — {START_DATE} to {END_DATE} ({T} months)")
print(f"{'='*55}")
print(f"{'Source':<15} {'Contribution':>14} {'% of Total':>12}")
print("─" * 45)

for k, v in contribs.items():
    pct = v / total_ret * 100 if total_ret != 0 else 0
    direction = '▲' if v > 0 else '▼'
    print(f"  {k:<14} {v*100:>+10.1f}%    {pct:>+8.1f}%  {direction}")

print("─" * 45)
print(f"  {'Model Total':<14} {model_total*100:>+10.1f}%")
print(f"  {'Residual':<14} {residual*100:>+10.1f}%   (rounding / nonlinearity)")
print(f"  {'ACTUAL TOTAL':<14} {total_ret*100:>+10.1f}%")

# ── CELL 9 — ROLLING BETAS: TIME-VARYING FACTOR EXPOSURES ───────────

ROLL_WINDOW = 60

X_ff6 = sm.add_constant(F[FACTOR_SETS['FF6']])
rols  = RollingOLS(excess_P, X_ff6, window=ROLL_WINDOW).fit()
rolling_betas = rols.params.dropna()

print(f"Rolling {ROLL_WINDOW}-month betas computed: {len(rolling_betas)} observations")
print(f"\nMost recent factor loadings ({rolling_betas.index[-1].date()}):")
for col in rolling_betas.columns:
    if col != 'const':
        print(f"  {col:<10} {rolling_betas[col].iloc[-1]:>+.3f}")

# ── CELL 10 — STOCK-LEVEL INDIVIDUAL REGRESSIONS ────────────────────

def regress_individual_stocks(stock_rets, factors_df, factor_cols):
    rows = []
    for ticker in stock_rets.columns:
        common = factors_df.index.intersection(stock_rets.index)
        excess  = stock_rets.loc[common, ticker] - factors_df.loc[common, 'RF']
        X       = sm.add_constant(factors_df.loc[common, factor_cols])
        m       = sm.OLS(excess, X).fit()

        row = {'Ticker': ticker,
               'Alpha bps/yr': round(m.params['const'] * 12 * 10000),
               't(α)': round(m.tvalues['const'], 2),
               'R²': round(m.rsquared, 3)}
        for f in factor_cols:
            row[f] = round(m.params[f], 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index('Ticker')

print("Individual stock regressions (FF6)...\n")
stock_summary = regress_individual_stocks(stock_returns, F, FACTOR_SETS['FF6'])
print(stock_summary.to_string())

# ── CELL 11 — CHART 1: FACTOR LOADINGS COMPARISON ───────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Factor Loadings: CAPM vs FF3 vs FF6', fontsize=14, fontweight='bold', y=1.01)

all_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

for ax, (mname, model) in zip(axes, models.items()):
    in_model = [f for f in all_factors if f in model.params.index]
    betas    = [model.params[f] for f in in_model]
    tstats   = [model.tvalues[f] for f in in_model]
    colors   = [C['green'] if b > 0 else C['red'] for b in betas]

    bars = ax.barh(in_model, betas, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.2)

    for bar, t, b in zip(bars, tstats, betas):
        x_pos = b + (0.015 if b >= 0 else -0.015)
        ha    = 'left' if b >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f't={t:.1f}', va='center', ha=ha, fontsize=9, color='#333')

    ax.axvline(0, color='black', linewidth=0.8)
    alpha_bps = model.params['const'] * 12 * 10000
    ax.set_title(f'{mname}\nα = {alpha_bps:+.0f} bps/yr  R²={model.rsquared:.2f}', fontsize=11)
    ax.set_xlabel('Factor Beta')

plt.tight_layout()
plt.savefig('1_factor_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 1_factor_loadings.png")

# ── CELL 12 — CHART 2: RETURN ATTRIBUTION WATERFALL ─────────────────

labels = list(contribs.keys()) + ['Total']
values = [v * 100 for v in contribs.values()]
total_val = total_ret * 100

fig, ax = plt.subplots(figsize=(12, 6))

running = 0
for i, (label, val) in enumerate(zip(labels[:-1], values)):
    bottom = min(running, running + val)
    height = abs(val)
    color  = C['green'] if val >= 0 else C['red']
    bar = ax.bar(i, height, bottom=bottom, color=color, alpha=0.85,
                 edgecolor='white', linewidth=1.5, width=0.6)
    ax.text(i, bottom + height / 2, f'{val:+.1f}%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    running += val

ax.bar(len(labels) - 1, abs(total_val), bottom=0 if total_val >= 0 else total_val,
       color=C['blue'], alpha=0.85, edgecolor='white', linewidth=1.5, width=0.6)
ax.text(len(labels) - 1, abs(total_val) / 2, f'{total_val:+.1f}%',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_title('Return Attribution (Arithmetic) — Period Return by Source', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Return Contribution (%)')

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=C['green'], label='Positive factor contribution'),
    Patch(facecolor=C['red'],   label='Negative factor contribution'),
    Patch(facecolor=C['blue'],  label='Total realized return'),
], loc='upper right')

plt.tight_layout()
plt.savefig('2_attribution_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 2_attribution_waterfall.png")

# ── CELL 13 — CHART 3: ROLLING BETAS OVER TIME ──────────────────────

factors_to_plot = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
colors_list     = [C['blue'], C['orange'], C['red'], C['green'], C['purple'], C['grey']]

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()
fig.suptitle(f'Rolling {ROLL_WINDOW}-Month Factor Exposures — Time-Varying Bets',
             fontsize=13, fontweight='bold')

for ax, factor, color in zip(axes, factors_to_plot, colors_list):
    if factor not in rolling_betas.columns:
        ax.set_visible(False)
        continue

    series = rolling_betas[factor]
    ax.plot(series.index, series, color=color, linewidth=1.8)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.fill_between(series.index, series, 0,
                    where=(series > 0), alpha=0.12, color=C['green'])
    ax.fill_between(series.index, series, 0,
                    where=(series < 0), alpha=0.12, color=C['red'])

    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'),
               alpha=0.08, color='black', label='COVID crash')
    ax.set_title(f'{factor} Loading', fontsize=11)
    ax.set_ylabel('Beta')

axes[-1].xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter('%Y'))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('3_rolling_betas.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 3_rolling_betas.png")

# ── CELL 14 — CHART 4: EQUITY CURVES + DRAWDOWN ─────────────────────

X_pred   = sm.add_constant(F[FACTOR_SETS['FF6']])
predicted_excess = models['FF6'].predict(X_pred)
predicted_returns = pd.Series(predicted_excess + F['RF'].values,
                               index=P.index, name='FF6 Predicted')

def cumwealth(ret):
    return (1 + ret).cumprod()

cum_P    = cumwealth(P)
cum_B    = cumwealth(B)
cum_pred = cumwealth(predicted_returns)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                 gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(cum_P.index,    cum_P,    color=C['blue'],   linewidth=2.2,  label='Portfolio')
ax1.plot(cum_B.index,    cum_B,    color=C['grey'],   linewidth=1.5,  linestyle='--', label=BENCHMARK)
ax1.plot(cum_pred.index, cum_pred, color=C['orange'], linewidth=1.5,  linestyle='-.', label='FF6 Predicted')

ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.2f}'))
ax1.set_title('Cumulative Wealth: Portfolio vs. SPY vs. Factor Model Prediction',
               fontsize=13, fontweight='bold')
ax1.set_ylabel('Growth of $1.00')
ax1.legend(loc='upper left')

ax1.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'),
            alpha=0.1, color='red', label='COVID')

roll_max = cum_P.cummax()
drawdown = (cum_P - roll_max) / roll_max

ax2.fill_between(drawdown.index, drawdown * 100, 0, color=C['red'], alpha=0.45)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Date')

stats = (f"Sharpe: {p_sr:.2f}  |  Ann. Return: {p_ret:.1%}  "
         f"|  Max Drawdown: {p_dd:.1%}  |  Ann. Alpha: {models['FF6'].params['const']*12*10000:+.0f} bps")
fig.text(0.5, -0.01, stats, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9))

plt.tight_layout()
plt.savefig('4_equity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 4_equity_curves.png")

# ── CELL 15 — CHART 5: FACTOR CORRELATION HEATMAP ───────────────────

corr = F[FACTOR_SETS['FF6']].corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson Correlation')

labels_ff6 = FACTOR_SETS['FF6']
ax.set_xticks(range(len(labels_ff6)))
ax.set_yticks(range(len(labels_ff6)))
ax.set_xticklabels(labels_ff6, rotation=45, ha='right')
ax.set_yticklabels(labels_ff6)

for i in range(len(labels_ff6)):
    for j in range(len(labels_ff6)):
        val = corr.iloc[i, j]
        txt_color = 'white' if abs(val) > 0.55 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=txt_color)

ax.set_title('Fama-French Factor Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('5_factor_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 5_factor_correlation.png")

# ── CELL 16 — FINAL INVESTMENT MEMO ─────────────────────────────────

m = models['FF6']
alpha_bps = m.params['const'] * 12 * 10000

size_interp = ('large-cap tilt' if m.params['SMB'] < -0.1 else
               'small-cap tilt' if m.params['SMB'] >  0.1 else 'size-neutral')
value_interp = ('value tilt'  if m.params['HML'] >  0.1 else
                'growth tilt' if m.params['HML'] < -0.1 else 'blend')
alpha_sig = ('SIGNIFICANT (|t| > 2.0)'
             if abs(m.tvalues['const']) > 2
             else 'NOT significant (|t| < 2.0)')

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            FACTOR ATTRIBUTION — INVESTMENT MEMO                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Universe : {', '.join(TICKERS)}
║  Period   : {F.index[0].date()} → {F.index[-1].date()} ({T} months)
╠══════════════════════════════════════════════════════════════════╣
║  PERFORMANCE SUMMARY                                             ║
║  Ann. Return          {p_ret*100:>8.2f}%   (SPY: {b_ret*100:.2f}%)
║  Ann. Volatility      {p_vol*100:>8.2f}%   (SPY: {b_vol*100:.2f}%)
║  Sharpe Ratio         {p_sr:>8.2f}    (SPY: {b_sr:.2f})
║  Max Drawdown         {p_dd*100:>8.2f}%
╠══════════════════════════════════════════════════════════════════╣
║  FACTOR EXPOSURES (FF6 Model)                                    ║
║  Ann. Alpha           {alpha_bps:>+8.0f} bps/yr  [{alpha_sig}]
║  Market Beta (Mkt-RF) {m.params['Mkt-RF']:>+8.3f}
║  Size Tilt    (SMB)   {m.params['SMB']:>+8.3f}   [{size_interp}]
║  Value Tilt   (HML)   {m.params['HML']:>+8.3f}   [{value_interp}]
║  Profitability(RMW)   {m.params['RMW']:>+8.3f}
║  Investment   (CMA)   {m.params['CMA']:>+8.3f}
║  Momentum     (Mom)   {m.params['Mom']:>+8.3f}
║  R² (variance explained) {m.rsquared*100:>5.1f}%
╠══════════════════════════════════════════════════════════════════╣
║  BOTTOM LINE                                                     ║
║  • {alpha_bps:+.0f} bps of annualized alpha vs. 6-factor benchmark
║  • Alpha is {alpha_sig}
║  • {m.rsquared*100:.1f}% of return variance explained by systematic factors
║  • Remaining {(1-m.rsquared)*100:.1f}% is idiosyncratic (stock-specific risk)
╚══════════════════════════════════════════════════════════════════╝
""")
