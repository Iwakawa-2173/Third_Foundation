"""
Клиодинамическая модель расчета и визуализации индексов SDT (PSI, EMP, MMP)
для России и Украины перед кризисом 2022 года.
Разработано в рамках проверки структурно-демографической теории П. Турчина.
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Входные исторические макроданные (2011-2021)
data_russia = {
    'Year': [2011, 2013, 2015, 2017, 2019, 2021],
    'w_inv': [1.85, 1.90, 2.10, 2.15, 2.20, 2.31],
    'urb_share': [0.738, 0.740, 0.741, 0.743, 0.746, 0.748],
    'youth_share': [0.151, 0.142, 0.128, 0.115, 0.106, 0.101],
    'elite_excess': [1.10, 1.25, 1.40, 1.65, 1.80, 2.15],
    'state_debt_share': [0.35, 0.38, 0.45, 0.42, 0.44, 0.52]
}

data_ukraine = {
    'Year': [2011, 2013, 2015, 2017, 2019, 2021],
    'w_inv': [2.10, 2.25, 2.65, 2.50, 2.45, 2.55],
    'urb_share': [0.687, 0.691, 0.692, 0.694, 0.696, 0.698],
    'youth_share': [0.148, 0.143, 0.131, 0.118, 0.108, 0.098],
    'elite_excess': [1.30, 1.55, 1.85, 2.10, 2.40, 2.85],
    'state_debt_share': [0.85, 0.98, 1.75, 1.60, 1.25, 1.15]
}

# Инициализация датафреймов
df_ru = pd.DataFrame(data_russia)
df_ua = pd.DataFrame(data_ukraine)

def calculate_cliodynamics(df):
    """Рассчитывает компоненты SDT и финальный индекс PSI."""
    df['MMP'] = df['w_inv'] * df['urb_share'] * df['youth_share']
    df['EMP'] = df['elite_excess']
    df['SFB'] = df['state_debt_share']
    df['PSI'] = df['MMP'] * df['EMP'] * df['SFB']
    return df

# Проведение расчетов
df_ru = calculate_cliodynamics(df_ru)
df_ua = calculate_cliodynamics(df_ua)

# Вывод результатов в консоль
print("--- КЛИОДИНАМИЧЕСКИЕ ПОКАЗАТЕЛИ: РОССИЯ ---")
print(df_ru[['Year', 'MMP', 'EMP', 'SFB', 'PSI']].round(3).to_string(index=False))

print("\n--- КЛИОДИНАМИЧЕСКИЕ ПОКАЗАТЕЛИ: УКРАИНА ---")
print(df_ua[['Year', 'MMP', 'EMP', 'SFB', 'PSI']].round(3).to_string(index=False))

# 2. Визуализация результатов (Три отдельных графика в одном окне)
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# Стилизация линий
line_style_ru = dict(marker='o', linewidth=2.5, color='blue', label='Россия')
line_style_ua = dict(marker='s', linewidth=2.5, color='orange', label='Украина')

# График 1: Потенциал массовой мобилизации (MMP)
axes[0].plot(df_ru['Year'], df_ru['MMP'], **line_style_ru)
axes[0].plot(df_ua['Year'], df_ua['MMP'], **line_style_ua)
axes[0].set_title('Потенциал массовой мобилизации ($MMP$)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Значение MMP', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# График 2: Перепроизводство и конкуренция элит (EMP)
axes[1].plot(df_ru['Year'], df_ru['EMP'], **line_style_ru)
axes[1].plot(df_ua['Year'], df_ua['EMP'], **line_style_ua)
axes[1].set_title('Перепроизводство и конкуренция элит ($EMP$)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Значение EMP', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# График 3: Индекс политического напряжения (PSI)
axes[2].plot(df_ru['Year'], df_ru['PSI'], **line_style_ru)
axes[2].plot(df_ua['Year'], df_ua['PSI'], **line_style_ua)
axes[2].set_title('Итоговый индекс политического напряжения ($PSI$)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Год', fontsize=11)
axes[2].set_ylabel('Значение PSI', fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()

# Автоматическое выравнивание отступов
plt.tight_layout()
plt.show()
