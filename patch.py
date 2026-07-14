import re

with open('factor_zoo.py', 'r') as f:
    content = f.read()

# Fix ann_stats
content = content.replace(
    '    ann_ret = (1 + ret_series.mean()) ** 12 - 1',
    '    ann_ret = (1 + cum) ** (12 / len(ret_series)) - 1 if len(ret_series) > 0 else 0'
)
content = content.replace(
    '    sharpe  = excess.mean() / ret_series.std() * np.sqrt(12)',
    '    sharpe  = excess.mean() / excess.std() * np.sqrt(12)'
)

# Fix decompose_attribution
old_decompose = """def decompose_attribution(model, factors_df, factor_cols, total_months):
    params       = model.params
    factor_means = factors_df[factor_cols].mean()

    contribs = {'Alpha': params['const'] * total_months}
    for f in factor_cols:
        contribs[f] = params[f] * factor_means[f] * total_months

    return contribs

T = len(F)
contribs = decompose_attribution(models['FF6'], F, FACTOR_SETS['FF6'], T)

total_cum   = (1 + P).prod() - 1
model_total = sum(contribs.values())
residual    = total_cum - model_total"""

new_decompose = """def decompose_attribution(model, factors_df, factor_cols, total_months):
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
contribs['Residual'] = residual"""

content = content.replace(old_decompose, new_decompose)

content = content.replace('total_cum', 'total_ret')

content = content.replace(
    "ax.set_title('Return Attribution — Cumulative Period Return by Source', fontsize=13, fontweight='bold')",
    "ax.set_title('Return Attribution (Arithmetic) — Period Return by Source', fontsize=13, fontweight='bold')"
)

with open('factor_zoo.py', 'w') as f:
    f.write(content)

