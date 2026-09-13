import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def get_geopolitical_dataset():
    years = list(range(1953, 2027, 1))
    hist_data = []
    for y in years:
        # 1. Базовые демографические тренды (Ретроспектива)
        if y <= 1991: urb = 0.45 + (0.66 - 0.45) * (y - 1953) / (1991 - 1953)
        else: urb = 0.66 + (0.75 - 0.66) * (y - 1991) / (2026 - 1991)
        
        youth_base = 0.15 + 0.015 * np.sin((y - 1953) / 4.5) if y <= 1991 else 0.145 - 0.05 * (y - 1991) / 35
        youth = max(0.08, youth_base + 0.005 * np.cos(y / 2.0))
        
        # 2. Траектория инверсированной зарплаты w^(-1)
        if y <= 1970: w_inv = 1.2 + 0.08 * np.sin((y - 1953) / 5)
        elif y <= 1985: w_inv = 1.28 + 0.22 * (y - 1970) / 15
        elif y <= 1991: w_inv = 1.5 + 1.0 * (y - 1985) / 6
        elif y <= 1998: w_inv = 2.5 + 0.6 * (y - 1991) / 7
        elif y <= 2011: w_inv = 3.1 - 1.25 * (y - 1998) / 13
        else: w_inv = 1.85 + 0.65 * (y - 2011) / 15
        
        # 3. Нарастание избытка элит EMP
        if y <= 1980: emp = 1.0 + 0.5 * (y - 1953) / 27
        elif y <= 1991: emp = 1.5 + 1.4 * (y - 1980) / 11
        elif y <= 2000: emp = 2.9 - 1.7 * (y - 1991) / 9
        elif y <= 2011: emp = 1.2 - 0.1 * (y - 2000) / 11
        else: emp = 1.1 + 1.4 * (y - 2011) / 15
        
        # 4. Фискальное давление государства SFB
        if y <= 1985: sfb = 0.3 + 0.1 * (y - 1953) / 32
        elif y <= 1991: sfb = 0.4 + 1.55 * (y - 1985) / 6
        elif y <= 1998: sfb = 1.95 + 0.2 * (y - 1991) / 7
        elif y <= 2011: sfb = 2.15 - 1.8 * (y - 1998) / 13
        else: sfb = 0.35 + 0.45 * (y - 2011) / 15
        
        # 5. Внешние геополитические потоки
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
        
        # 6. Введение параболы информационного обмена
        if y < 2014: i_flow = 0.5
        elif y < 2022: i_flow = 0.4
        else: i_flow = 0.3
        
        m_cog = (4.0 * 1.0 * i_flow * (1.0 - i_flow)) / (1.0**2)
        hist_data.append([y, w_inv, urb, youth, emp, sfb, oil, sanctions, usa, eu, china, xi, i_flow, m_cog])
        
    columns = ['Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB', 'Oil', 'Sanctions', 'USA', 'EU', 'China', 'Xi', 'I_flow', 'M_cog']
    df = pd.DataFrame(hist_data, columns=columns)
    df['MMP'] = df['w_inv'] * df['Urb'] * df['Youth']
    df['PSI'] = (df['MMP'] / df['M_cog']) * df['EMP'] * df['SFB'] * df['Xi']
    return df

def execute_final_simulation():
    # Загрузка и подготовка исторических данных
    df_hist = get_geopolitical_dataset()
    future_years = np.arange(2027, 2057, 1)
    
    X_train = df_hist[['Year']].values
    X_future = future_years.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)
    
    # Обучение многослойного перцептрона ИИ (128, 64, 32 нейрона)
    nn = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='tanh', solver='lbfgs', max_iter=10000, random_state=42)
    df_pred = pd.DataFrame({'Year': future_years})
    
    targets = ['w_inv', 'EMP', 'SFB', 'Oil', 'Sanctions', 'USA', 'EU', 'China']
    for t in targets:
        nn.fit(X_train_scaled, df_hist[t].values)
        df_pred[t] = nn.predict(X_future_scaled)
        
    # Экстраполяция жестких демографических компонентов
    df_pred['Urb'] = 0.75 + (0.77 - 0.75) * (future_years - 2026) / (2056 - 2026)
    df_pred['Youth'] = 0.09 - (0.09 - 0.071) * (future_years - 2026) / (2056 - 2026)
    df_pred['Xi'] = (df_pred['Oil'] * 0.25) + (df_pred['Sanctions'] * 0.25) + (df_pred['USA'] * 0.20) + (df_pred['EU'] * 0.15) + (df_pred['China'] * 0.15)
    df_pred['MMP'] = df_pred['w_inv'] * df_pred['Urb'] * df_pred['Youth']
    
    # Прогноз Сценария: Опускание железного занавеса (падение I_flow до минимальных 0.1 к 2040 году)
    df_pred['I_flow'] = 0.3 - (0.3 - 0.1) * (future_years - 2026) / (2040 - 2026)
    df_pred['I_flow'] = df_pred['I_flow'].clip(lower=0.1)
    df_pred['M_cog'] = (4.0 * 1.0 * df_pred['I_flow'] * (1.0 - df_pred['I_flow'])) / (1.0**2)
    df_pred['PSI'] = (df_pred['MMP'] / df_pred['M_cog']) * df_pred['EMP'] * df_pred['SFB'] * df_pred['Xi']
    
    # Объединение в один датасет для вывода полного сквозного лога
    df_total = pd.concat([df_hist, df_pred], ignore_index=True)
    
    # Вывод полного лога в консоль
    print("="*105)
    print(f"{'Год':<5} | {'MMP':<6} | {'EMP':<6} | {'SFB':<6} | {'Xi (Кси)':<8} | {'I_flow':<7} | {'M_cog':<6} | {'PSI_geo':<8} | {'Статус Системы'}")
    print("="*105)
    
    USSR_MARKER = 1.4730
    breakthrough_year = None
    
    for _, row in df_total.iterrows():
        status = "СТАБИЛЬНО"
        if row['Year'] == 1991: status = "КОЛЛАПС СССР"
        elif row['Year'] == 1998: status = "ДЕФОЛТ РФ"
        elif row['Year'] == 2022: status = "ГЕОПОЛИТИЧЕСКИЙ ИЗЛОМ"
        elif row['PSI'] > USSR_MARKER:
            status = "КРИЗИСНЫЙ ПРОБОЙ 🚨"
            if breakthrough_year is None and row['Year'] > 2026:
                breakthrough_year = int(row['Year'])
                
        print(f"{int(row['Year']):<5} | {row['MMP']:<6.3f} | {row['EMP']:<6.3f} | {row['SFB']:<6.3f} | {row['Xi']:<8.3f} | {row['I_flow']:<7.3f} | {row['M_cog']:<6.3f} | {row['PSI']:<8.4f} | {status}")
    
    print("="*105)
    print(f"ИСХОДНАЯ ТОЧКА ПРОЛОМА ИИ: 2046 ГОД")
    print(f"НОВАЯ ТОЧКА СИСТЕМНОГО ПРОЛОМА (С УЧЕТОМ ЦЕНЗУРЫ И НЕВЕДЕНИЯ): {breakthrough_year} ГОД!")
    print("="*105)
    
    # --- СТРОИТЕЛЬСТВО 4-ПАНЕЛЬНОГО ГРАФИКА ВЫСОКОГО РАЗРЕШЕНИЯ ---
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, dpi=120)
    
    style_h = dict(color='#111111', linewidth=2.5, label='Историческая ретроспектива (1953–2026)')
    style_p = dict(color='#7E1E9C', linestyle='--', linewidth=2.2, marker='o', markersize=3, label='Модифицированный ИИ-прогноз с цензурой (2027–2056)')
    
    metrics = [
        {'key': 'MMP', 'title': r'Потенциал массовой мобилизации ($MMP$)', 'ylim': [0.05, 0.45]},
        {'key': 'EMP', 'title': r'Перепроизводство элит ($EMP$)', 'ylim': [0.5, 4.8]},
        {'key': 'M_cog', 'title': r'Когнитивная мотивация масс ($M_{cog}$)', 'ylim': [0.2, 1.1]},
        {'key': 'PSI', 'title': r'Итоговый индекс геополитического напряжения ($PSI_{geo}$)', 'ylim': [-0.1, 4.5]}
    ]
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        k = m['key']
        ax.plot(df_hist['Year'], df_hist[k], **style_h)
        ax.plot(df_pred['Year'], df_pred[k], **style_p)
        
        # Маркировка исторических периодов застоя и стабильности
        ax.axvspan(1953, 1980, color='#2ECC71', alpha=0.04)
        ax.axvspan(1986, 1999, color='#E74C3C', alpha=0.04)
        ax.axvspan(2022, 2026, color='#F1C40F', alpha=0.06)
        
        # Разделительные веховые вертикали
        ax.axvline(x=1991, color='#C0392B', linestyle=':', alpha=0.6)
        ax.axvline(x=2022, color='#7E1E9C', linestyle='-.', alpha=0.5)
        ax.axvline(x=2026, color='#7F8C8D', linestyle='-', alpha=0.5)
        if breakthrough_year:
            ax.axvline(x=breakthrough_year, color='#D35400', linestyle='-', linewidth=2, alpha=0.8)
            
        ax.set_title(m['title'], fontsize=12, fontweight='bold', color='#2C3E50', pad=6)
        ax.set_ylim(m['ylim'])
        
    axes[3].set_xlim(1953, 2060)
    axes[3].set_xticks(np.arange(1953, 2061, 5))
    axes[3].set_xlabel('Год клиодинамического исследования', fontsize=12, labelpad=10, color='#2C3E50')
    
    # Добавление текстовых маркеров на график PSI
    axes[3].text(1991.5, 1.65, 'Коллапс СССР (1.48)', color='#C0392B', fontsize=9, fontweight='bold')
    axes[3].text(2022.5, 2.25, 'Излом 2022 г.', color='#7E1E9C', fontsize=9, fontweight='bold')
    if breakthrough_year:
        axes[3].text(breakthrough_year + 0.5, 1.60, f'Сдвиг пролома на {breakthrough_year} г.!', color='#D35400', fontsize=10, fontweight='bold')
        
    fig.suptitle('Модифицированная ИИ-модель PSI_geo: Интеграция когнитивного фильтра масс', fontsize=14, fontweight='bold', y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    execute_final_simulation()
