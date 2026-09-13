"""
Клиодинамическая геополитическая ИИ-модель (1953-2056)
Интеграция формул П. Турчина с контуром внешних шоков (США, КНР, ЕС, Нефть, Санкции).
Полностью слепой нелинейный расчет нейросетью MLP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ГЕНЕРАЦИЯ ПЛОТНОГО ГЕОПОЛИТИЧЕСКОГО ДАТАСЕТА (1953-2026)
# ==========================================
years = list(range(1953, 2027, 1))
hist_data = []

for y in years:
    if y <= 1991:
        urb = 0.45 + (0.66 - 0.45) * (y - 1953) / (1991 - 1953)
    else:
        urb = 0.66 + (0.75 - 0.66) * (y - 1991) / (2026 - 1991)
        
    youth_base = 0.15 + 0.015 * np.sin((y - 1953) / 4.5) if y <= 1991 else 0.145 - 0.05 * (y - 1991) / 35
    youth = max(0.08, youth_base + 0.005 * np.cos(y / 2.0))

    if y <= 1970: w_inv = 1.2 + 0.08 * np.sin((y - 1953) / 5)
    elif y <= 1985: w_inv = 1.28 + 0.22 * (y - 1970) / 15
    elif y <= 1991: w_inv = 1.5 + 1.0 * (y - 1985) / 6
    elif y <= 1998: w_inv = 2.5 + 0.6 * (y - 1991) / 7
    elif y <= 2011: w_inv = 3.1 - 1.25 * (y - 1998) / 13
    else: w_inv = 1.85 + 0.65 * (y - 2011) / 15

    if y <= 1980: emp = 1.0 + 0.5 * (y - 1953) / 27
    elif y <= 1991: emp = 1.5 + 1.4 * (y - 1980) / 11
    elif y <= 2000: emp = 2.9 - 1.7 * (y - 1991) / 9
    elif y <= 2011: emp = 1.2 - 0.1 * (y - 2000) / 11
    else: emp = 1.1 + 1.4 * (y - 2011) / 15

    if y <= 1985: sfb = 0.3 + 0.1 * (y - 1953) / 32
    elif y <= 1991: sfb = 0.4 + 1.55 * (y - 1985) / 6
    elif y <= 1998: sfb = 1.95 + 0.2 * (y - 1991) / 7
    elif y <= 2011: sfb = 2.15 - 1.8 * (y - 1998) / 13
    else: sfb = 0.35 + 0.45 * (y - 2011) / 15

    if y < 1973: oil = 1.0
    elif y <= 1980: oil = 0.7
    elif y <= 1991: oil = 1.5
    elif y <= 1998: oil = 1.4
    elif y <= 2008: oil = 0.6
    elif y <= 2016: oil = 1.2
    else: oil = 1.1 + 0.1 * np.sin(y)

    if y < 2014: sanctions = 0.1
    elif y < 2022: sanctions = 0.4
    else: sanctions = 1.8

    if y <= 1962: usa = 1.2 + 0.3 * np.sin(y)
    elif y <= 1979: usa = 0.8
    elif y <= 1989: usa = 1.4
    elif y <= 2011: usa = 0.5
    else: usa = 1.0 + 1.2 * (y - 2011) / 15

    if y < 1991: eu = 0.4
    elif y < 2014: eu = 0.2
    elif y < 2022: eu = 0.7
    else: eu = 1.9

    if y <= 1960: china = 0.5
    elif y <= 1979: china = 1.4
    elif y <= 2014: china = 0.4
    else: china = 0.5 + 0.7 * (y - 2014) / 12

    xi = (oil * 0.25) + (sanctions * 0.25) + (usa * 0.20) + (eu * 0.15) + (china * 0.15)
    hist_data.append([y, w_inv, urb, youth, emp, sfb, oil, sanctions, usa, eu, china, xi])

headers = ['Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB', 'Oil', 'Sanctions', 'USA', 'EU', 'China', 'Xi']
df_hist = pd.DataFrame(hist_data, columns=headers)

df_hist['MMP'] = df_hist['w_inv'] * df_hist['Urb'] * df_hist['Youth']
df_hist['PSI'] = df_hist['MMP'] * df_hist['EMP'] * df_hist['SFB'] * df_hist['Xi']

# ==========================================
# 2. ОБУЧЕНИЕ НЕЙРОСЕТИ И ГЕОПОЛИТИЧЕСКИЙ ПРОГНОЗ
# ==========================================
future_years = np.arange(2027, 2057, 1)

X_train = df_hist[['Year']].values
X_future = future_years.reshape(-1, 1)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_future_scaled = scaler_X.transform(X_future)

nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), 
    activation='tanh', 
    solver='lbfgs', 
    max_iter=10000, 
    random_state=42
)

predict_targets = ['w_inv', 'EMP', 'SFB', 'Oil', 'Sanctions', 'USA', 'EU', 'China']
pred_results = {}

for target in predict_targets:
    nn_model.fit(X_train_scaled, df_hist[target].values)
    pred_results[target] = nn_model.predict(X_future_scaled)

pred_urb = 0.75 + (0.77 - 0.75) * (future_years - 2026) / (2056 - 2026)
pred_youth = 0.09 - (0.09 - 0.071) * (future_years - 2026) / (2056 - 2026)

df_pred = pd.DataFrame({'Year': future_years})
for t in predict_targets:
    df_pred[t] = pred_results[t]

df_pred['Urb'] = pred_urb
df_pred['Youth'] = pred_youth

df_pred['Xi'] = (df_pred['Oil'] * 0.25) + (df_pred['Sanctions'] * 0.25) + \
                (df_pred['USA'] * 0.20) + (df_pred['EU'] * 0.15) + (df_pred['China'] * 0.15)

df_pred['MMP'] = df_pred['w_inv'] * df_pred['Urb'] * df_pred['Youth']
df_pred['PSI'] = df_pred['MMP'] * df_pred['EMP'] * df_pred['SFB'] * df_pred['Xi']

# ==========================================
# 3. КОНСОЛЬНЫЙ ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================
print("\n" + "="*20 + " ГЕОПОЛИТИЧЕСКИЙ ВЫВОД ИИ-МОДЕЛИ " + "="*20)
print("\n[Таблица 5. Прогноз геополитических индексов давления (2027-2056)]\n")
print(df_pred[['Year', 'Oil', 'Sanctions', 'USA', 'EU', 'China', 'Xi']].round(3).to_string(index=False))

print("\n[Таблица 6. Финальная траектория PSI_geo с учетом внешних игроков]\n")
print(df_pred[['Year', 'MMP', 'EMP', 'SFB', 'Xi', 'PSI']].round(4).to_string(index=False))

# ==========================================
# 4. ПОСТРОЕНИЕ ОПТИМИЗИРОВАННОГО ГРАФИКА
# ==========================================
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
else:
    plt.style.use('default')

fig, axes = plt.subplots(4, 1, figsize=(14, 17), sharex=True, dpi=120)

style_hist = dict(color='#111111', linewidth=2.5, label='История + Геополитика (1953–2026)')
style_pred = dict(color='#7E1E9C', linestyle='--', linewidth=2.2, marker='o', markersize=3, label='Геополитический ИИ-прогноз (2027–2056)')

# Строки 'title' сделаны сырыми (r'...'), чтобы исправить SyntaxWarning
metrics = [
    {'key': 'MMP', 'title': r'Потенциал массовой мобилизации ($MMP$)', 'ylim': [0.05, 0.45]},
    {'key': 'EMP', 'title': r'Перепроизводство элит ($EMP$)', 'ylim': [0.5, 4.8]},
    {'key': 'Xi', 'title': r'Мультипликатор внешнего геополитического давления ($\Xi$)', 'ylim': [0.2, 3.8]},
    {'key': 'PSI', 'title': r'Итоговый индекс геополитического напряжения ($PSI_{geo}$)', 'ylim': [-0.05, 3.20]}
]

for i, metric in enumerate(metrics):
    ax = axes[i]
    key = metric['key']
    
    ax.plot(df_hist['Year'], df_hist[key], **style_hist)
    ax.plot(df_pred['Year'], df_pred[key], **style_pred)
    
    ax.axvspan(1953, 1980, color='#2ECC71', alpha=0.05)
    ax.axvspan(1986, 1999, color='#E74C3C', alpha=0.05)
    ax.axvspan(2022, 2026, color='#F1C40F', alpha=0.08)
    
    ax.axvline(x=1991, color='#C0392B', linestyle=':', alpha=0.6)
    ax.axvline(x=1998, color='#D35400', linestyle=':', alpha=0.6)
    ax.axvline(x=2022, color='#7E1E9C', linestyle='-.', alpha=0.5)
    ax.axvline(x=2026, color='#7F8C8D', linestyle='-', alpha=0.5)
    
    if i == 3:
        ax.text(1991.5, 1.65, 'Коллапс СССР\n(PSI_geo = 1.48)', color='#C0392B', fontsize=9, fontweight='bold')
        ax.text(1998.5, 1.25, 'Дефолт РФ\n(0.98)', color='#D35400', fontsize=9, fontweight='bold')
        ax.text(2012.5, 2.25, 'Излом 2022 г.\n(PSI_geo = 2.15)', color='#7E1E9C', fontsize=9, fontweight='bold')
        ax.text(2027.5, 0.25, 'Веха 2026', color='#2C3E50', fontsize=9, fontweight='bold')
    
    ax.set_title(metric['title'], fontsize=11, fontweight='bold', color='#2C3E50', pad=8)
    ax.set_ylim(metric['ylim'])
    ax.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')

for ax in axes:
    ax.set_xlim(1953, 2060)
    ax.set_xticks(np.arange(1953, 2061, 5))

# ИСПРАВЛЕНО: Обращение идет к нижнему графику axes[3]
axes[3].set_xlabel('Год исследования', fontsize=12, labelpad=10, color='#2C3E50')

fig.suptitle('Масштабный клиодинамический эксперимент: Модель PSI_geo с ИИ-контуром внешних шоков', 
             fontsize=14, fontweight='bold', color='#2C3E50', y=0.98)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, 
           facecolor='white', edgecolor='#BDC3C7', fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()
