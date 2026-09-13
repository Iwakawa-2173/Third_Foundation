import pandas as pd

# 1. Исходные расчетные данные ИИ-модели для ключевых точек (2046 и 2056 гг.)
# Формула: PSI_geo = MMP * EMP * SFB * Xi
data_baseline = {
    'Year': [2046, 2056],
    'MMP': [0.1856, 0.1827],
    'EMP': [3.4503, 3.6695],
    'SFB': [1.0756, 1.1206],
    'Xi':  [2.1661, 3.6585]
}

df = pd.DataFrame(data_baseline)

# Функция расчета базового индекса
def calc_baseline_psi(row):
    return row['MMP'] * row['EMP'] * row['SFB'] * row['Xi']

df['PSI_Baseline'] = df.apply(calc_baseline_psi, axis=1)

# Исторический маркер коллапса СССР (1991 год)
USSR_MARKER = 1.4730

# 2. Моделирование сценариев контрмер
# Сценарий 1: Снижение внешнего давления (Xi) на 20%
df['PSI_Scen1_External'] = df['MMP'] * df['EMP'] * df['SFB'] * (df['Xi'] * 0.80)

# Сценарий 2: Охлаждение элитной конкуренции (EMP) на 25%
df['PSI_Scen2_Internal'] = df['MMP'] * (df['EMP'] * 0.75) * df['SFB'] * df['Xi']

# Сценарий 3: Комбинированный (Xi -10%, EMP -15%)
df['PSI_Scen3_Combined'] = df['MMP'] * (df['EMP'] * 0.85) * df['SFB'] * (df['Xi'] * 0.90)

# 3. Вывод результатов анализа
print("=== РЕЗУЛЬТАТЫ СИМУЛЯЦИИ КОНТРМЕР ===")
print(f"Исторический порог прочности (СССР 1991): {USSR_MARKER}\n")

for i, year in enumerate(df['Year']):
    print(f"--- Прогноз на {year} год ---")
    print(f"Базовый прогноз ИИ (Без контрмер): {df['PSI_Baseline'].iloc[i]:.4f} " 
          f"({'ПРОБОЙ' if df['PSI_Baseline'].iloc[i] > USSR_MARKER else 'СТАБИЛЬНО'})")
    
    print(f"Сценарий 1 (Внешнее давление -20%): {df['PSI_Scen1_External'].iloc[i]:.4f} "
          f"({'ПРОБОЙ' if df['PSI_Scen1_External'].iloc[i] > USSR_MARKER else 'СТАБИЛЬНО'})")
    
    print(f"Сценарий 2 (Избыток элит -25%):     {df['PSI_Scen2_Internal'].iloc[i]:.4f} "
          f"({'ПРОБОЙ' if df['PSI_Scen2_Internal'].iloc[i] > USSR_MARKER else 'СТАБИЛЬНО'})")
    
    print(f"Сценарий 3 (Комбинированный):        {df['PSI_Scen3_Combined'].iloc[i]:.4f} "
          f"({'ПРОБОЙ' if df['PSI_Scen3_Combined'].iloc[i] > USSR_MARKER else 'СТАБИЛЬНО'})")
    print()
