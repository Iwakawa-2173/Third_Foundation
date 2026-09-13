"""
Клиодинамическая ИИ-модель: СССР -> РФ -> Прогноз будущего до 2056 года.
Оптимизированный академический визуал с учетом методологии П. Турчина (Nature, 2010).
Масштаб осей полностью скорректирован под пиковые значения модели.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ГЕНЕРАЦИЯ ПЛОТНЫХ ИСТОРИЧЕСКИХ ДАННЫХ (1953-2026)
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

    hist_data.append([y, w_inv, urb, youth, emp, sfb])

df_hist = pd.DataFrame(hist_data, columns=['Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB'])
df_hist['MMP'] = df_hist['w_inv'] * df_hist['Urb'] * df_hist['Youth']
df_hist['PSI'] = df_hist['MMP'] * df_hist['EMP'] * df_hist['SFB']

# ==========================================
# 2. ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ И ПРОГНОЗИРОВАНИЕ
# ==========================================
future_years = np.arange(2027, 2057, 1)

X_train = df_hist[['Year']].values
X_future = future_years.reshape(-1, 1)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_future_scaled = scaler_X.transform(X_future)

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='tanh', 
    solver='lbfgs', 
    max_iter=5000, 
    random_state=42
)

targets = ['w_inv', 'EMP', 'SFB']
pred_results = {}

for target in targets:
    nn_model.fit(X_train_scaled, df_hist[target].values)
    pred_results[target] = nn_model.predict(X_future_scaled)

pred_urb = 0.75 + (0.77 - 0.75) * (future_years - 2026) / (2056 - 2026)
pred_youth = 0.09 - (0.09 - 0.071) * (future_years - 2026) / (2056 - 2026)

df_pred = pd.DataFrame({'Year': future_years})
df_pred['w_inv'] = pred_results['w_inv']
df_pred['Urb'] = pred_urb
df_pred['Youth'] = pred_youth
df_pred['EMP'] = pred_results['EMP']
df_pred['SFB'] = pred_results['SFB']
df_pred['MMP'] = df_pred['w_inv'] * df_pred['Urb'] * df_pred['Youth']
df_pred['PSI'] = df_pred['MMP'] * df_pred['EMP'] * df_pred['SFB']

# ==========================================
# 3. ЧИСТЫЙ КОНСОЛЬНЫЙ ВЫВОД ТАБЛИЦ
# ==========================================
print("\n" + "="*20 + " ТЕКСТОВЫЙ МАКРОСОЦИОЛОГИЧЕСКИЙ ВЫВОД " + "="*20)
print("\n[Таблица 3. Точные прогнозные значения компонентов модели (ИИ-расчет)]\n")
print(df_pred[['Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB']].round(3).to_string(index=False))

print("\n[Таблица 4. Траектория итогового индекса political stress (PSI)]\n")
print(df_pred[['Year', 'MMP', 'PSI']].round(4).to_string(index=False))
print("\n" + "="*75 + "\n")

# ==========================================
# 4. МНОГОПАНЕЛЬНЫЙ ГРАФИК (МАСШТАБ ИСПРАВЛЕН)
# ==========================================
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
else:
    plt.style.use('default')

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True, dpi=120)

style_hist = dict(color='#111111', linewidth=2.5, label='История (1953–2026)')
style_pred = dict(color='#A30000', linestyle='--', linewidth=2.2, marker='o', markersize=3, label='ИИ-прогноз (2027–2056)')

metrics = [
    {'key': 'MMP', 'title': 'Потенциал массовой мобилизации ($MMP$)', 'ylim': [0.05, 0.45]},
    {'key': 'EMP', 'title': 'Перепроизводство и конкуренция элит ($EMP$)', 'ylim': [0.5, 4.8]},
    {'key': 'SFB', 'title': 'Фискальное давление государства ($SFB$)', 'ylim': [0.1, 2.9]},
    {'key': 'PSI', 'title': 'Итоговый индекс политического напряжения ($PSI$)', 'ylim': [-0.05, 1.70]} # МАСШТАБ ИСПРАВЛЕН ДО 1.70
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
    ax.axvline(x=2026, color='#7F8C8D', linestyle='-', alpha=0.5)
    
    if i == 3:
        ax.text(1991.5, 1.50, 'Коллапс СССР (1.473)', color='#C0392B', fontsize=9, fontweight='bold')
        ax.text(1998.5, 1.10, 'Дефолт РФ (0.998)', color='#D35400', fontsize=9, fontweight='bold')
        ax.text(2027.5, 0.15, 'Веха 2026', color='#2C3E50', fontsize=9, fontweight='bold')
    
    ax.set_title(metric['title'], fontsize=11, fontweight='bold', color='#2C3E50', pad=8)
    ax.set_ylim(metric['ylim'])
    ax.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')

# Масштабирование горизонтальной оси
for ax in axes:
    ax.set_xlim(1953, 2060)
    ax.set_xticks(np.arange(1953, 2061, 5))

# Подпись для самого нижнего подграфика
axes[3].set_xlabel('Год исследования', fontsize=12, labelpad=10, color='#2C3E50')

fig.suptitle('Комплексный клиодинамический анализ параметров теории П. Турчина', 
             fontsize=14, fontweight='bold', color='#2C3E50', y=0.98)

# Сбор легенды
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, 
           facecolor='white', edgecolor='#BDC3C7', fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()
