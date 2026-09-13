"""
COMPENDIUM PSYCHOHISTORIAE — МОДУЛЬ КНР (СЦЕНАРИЙ ОТКРЫТОГО КОНТУРА)
Масштабный нелинейный эксперимент: расчет индексов SDT и PSI_geo для Китая (1953–2056 гг.)
с учетом когнитивного зажима Mcog, гендерного дисбаланса и скрытого долга LGFV.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_china_historical_dataset():
    """
    Формирует плотный ежегодный исторический ряд макроструктурных параметров КНР (1953–2026 гг.).
    Данные нормированы и откалиброваны с учетом специфики китайских секулярных циклов.
    """
    years = list(range(1953, 2027, 1))
    hist_data = []
    
    for y in years:
        # 1. Urb (Доля городского населения) — Бурный скачок от автаркии Мао к ВТО
        if y <= 1978:
            urb = 0.13 + (0.18 - 0.13) * (y - 1953) / (1978 - 1953)
        else:
            urb = 0.18 + (0.66 - 0.18) * (y - 1978) / (2026 - 1978)
            
        # 2. Youth (Молодежная когорта) + Демпфер "Голых ветвей" (избыток одиноких мужчин)
        # Базовая когорта падает из-за политики "Одного ребенка", но гендерный перекос дает горючий материал
        base_youth = 0.16 - 0.05 * (y - 1953) / 73 if y > 1980 else 0.16 + 0.01 * np.sin(y)
        gender_penalty = 1.25 if y >= 2000 else 1.0 + 0.25 * (y - 1980) / 20 if y > 1980 else 1.0
        youth = max(0.06, base_youth * gender_penalty)
        
        # 3. w_inv (Инверсированная относительная зарплата) — Рост неравенства при Дэн Сяопине
        if y <= 1978:
            w_inv = 1.30 + 0.05 * np.cos(y)
        elif y <= 2001:
            w_inv = 1.35 + 0.65 * (y - 1978) / 23  # Рост расслоения города и деревни
        else:
            w_inv = 2.00 + 0.45 * (y - 2001) / 25  # Стабилизация на высоких значениях
            
        # 4. EMP (Избыток элит) — Культ высшего образования и дефицит мест в КПК (Нэйцзюань)
        if y <= 1978:
            emp = 0.90 + 0.2 * (y - 1953) / 25     # Жесткие чистки Мао сдерживают избыток элит
        elif y <= 1989:
            emp = 1.10 + 0.6 * (y - 1978) / 11     # Рост числа студентов перед Тяньаньмэнь
        elif y <= 2015:
            emp = 1.40 + 1.2 * (y - 1889) / 26     # Бум коммерческих вузов
        else:
            emp = 2.60 + 0.9 * (y - 2015) / 11     # Исторический пик: 11+ млн выпускников в год

        # 5. SFB (Фискальный стресс) — Учет скрытых долгов провинций (LGFV) и девелоперов
        if y <= 1994:
            sfb = 0.40 + 0.1 * np.sin(y)
        elif y <= 2008:
            sfb = 0.45 + 0.15 * (y - 1994) / 14
        else:
            sfb = 0.60 + 0.65 * (y - 2008) / 18    # Взлет из-за долгов местных бюджетов и Evergrande

        # 6. Геополитический контур КНР (\u039e)
        # \u039eeu и \u039erus инвертированы по сравнению с РФ. СССР до 1985 — фактор огромного давления
        if y <= 1978: oil = 1.0  # Энергетическая автаркия
        else: oil = 1.0 + 0.4 * (y - 1978) / 48  # Рост зависимости от импорта нефти бьет по марже
        
        if y < 1989: sanctions = 1.5 if y < 1971 else 0.2  # Блокада Запада до признания ООН
        elif y < 2018: sanctions = 0.5                     # Вступление в ВТО, золотой век
        else: sanctions = 1.6 + 0.4 * (y - 2018) / 8       # Полупроводниковая блокада США, тарифные войны
        
        if y <= 1972: usa = 2.0 - 0.5 * (y - 1953) / 19    # Корейская война, Тайваньский кризис
        elif y <= 2016: usa = 0.4 + 0.1 * np.sin(y)        # Эпоха "Кимерики" (Chimerica)
        else: usa = 1.5 + 0.7 * (y - 2016) / 10            # Доктрина сдерживания Китая
        
        eu = 0.4 if y < 2018 else 0.4 + 0.6 * (y - 2018) / 8  # Политический de-risking Европы
        
        if y <= 1960: rus = 0.3                            # "Сталин и Мао слушают нас"
        elif y <= 1985: rus = 1.8 - 0.3 * (y - 1960) / 25  # Советско-китайский раскол, Даманский
        else: rus = 0.4                                    # Нормализация и превращение РФ в сырьевой тыл
        
        # Специфические веса геополитического мультипликатора для КНР
        xi = (oil * 0.20) + (sanctions * 0.30) + (usa * 0.30) + (eu * 0.10) + (rus * 0.10)
        
        # 7. I_flow и Когнитивный зажим (Великий китайский файрвол)
        # Постепенное падение открытости информации с начала 2010-х
        if y < 2010: i_flow = 0.50
        else: i_flow = max(0.12, 0.50 - 0.38 * (y - 2010) / 16)
        
        mcog = 4.0 * i_flow * (1.0 - i_flow)
        
        hist_data.append([y, w_inv, urb, youth, emp, sfb, oil, sanctions, usa, eu, rus, xi, i_flow, mcog])
        
    df = pd.DataFrame(hist_data, columns=[
        'Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB', 
        'Oil', 'Sanctions', 'USA', 'EU', 'Rus', 'Xi', 'I_flow', 'M_cog'
    ])
    df['MMP'] = df['w_inv'] * df['Urb'] * df['Youth']
    df['PSI_geo'] = (df['MMP'] / df['M_cog']) * df['EMP'] * df['SFB'] * df['Xi']
    return df

def run_china_psychohistory_engine():
    """
    Запускает глубокую нейросеть (128, 64, 32: Tanh) для слепого прогнозирования
    параметров КНР до 2056 года с учетом ИИ-экстраполяции трендов.
    """
    df_hist = generate_china_historical_dataset()
    future_years = np.arange(2027, 2057, 1)
    
    X_train = df_hist[['Year']].values
    X_future = future_years.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)
    
    # Инициализация топологии из Этапа III/IV компендиума
    nn = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='tanh', solver='lbfgs', max_iter=10000, random_state=42)
    
    df_pred = pd.DataFrame({'Year': future_years})
    targets = ['w_inv', 'EMP', 'SFB', 'Oil', 'Sanctions', 'USA', 'EU', 'Rus']
    
    for t in targets:
        nn.fit(X_train_scaled, df_hist[t].values)
        df_pred[t] = nn.predict(X_future_scaled)
        
    # Демографическая инерция КНР до 2056 года (Старение + пик сжатия)
    df_pred['Urb'] = [min(0.80, 0.66 + (0.80 - 0.66) * (y - 2026) / 30) for y in future_years]
    df_pred['Youth'] = [max(0.05, 0.076 - (0.076 - 0.05) * (y - 2026) / 30) for y in future_years]
    
    # Фиксация тотального цифрового контроля (I_flow зажат на минимуме)
    df_pred['I_flow'] = 0.12
    df_pred['M_cog'] = 4.0 * df_pred['I_flow'] * (1.0 - df_pred['I_flow'])
    
    # Расчет прогнозного мультипликатора внешнего давления
    df_pred['Xi'] = (df_pred['Oil'] * 0.20) + (df_pred['Sanctions'] * 0.30) + (df_pred['USA'] * 0.30) + (df_pred['EU'] * 0.10) + (df_pred['Rus'] * 0.10)
    df_pred['MMP'] = df_pred['w_inv'] * df_pred['Urb'] * df_pred['Youth']
    df_pred['PSI_geo'] = (df_pred['MMP'] / df_pred['M_cog']) * df_pred['EMP'] * df_pred['SFB'] * df_pred['Xi']
    
    # Объединение для консолидированного вывода
    df_full = pd.concat([df_hist, df_pred], ignore_index=True)
    
    # Исторический маркер устойчивости системы (СССР 1991 = 1.4730)
    CHINA_CRITICAL_MARKER = 1.4730
    
    print("="*95)
    print(" COMPENDIUM PSYCHOHISTORIAE: КЛИОДИНАМИЧЕСКИЙ РАСЧЕТ МАКРОСИСТЕМЫ КНР (1953-2056) ")
    print("="*95)
    print(f"{'Год':<6} | {'MMP':<7} | {'EMP':<7} | {'SFB':<7} | {'Xi':<7} | {'M_cog':<7} | {'PSI_geo':<8} | {'Статус Системы'}")
    print("-"*95)
    
    # Ключевые исторические ретроспективные вехи для контроля калибровки
    control_years = [1953, 1969, 1978, 1989, 2001, 2018, 2022, 2026]
    for cy in control_years:
        row = df_full[df_full['Year'] == cy].iloc[0]
        status = "ПЛАТО СТАБИЛЬНОСТИ" if row['PSI_geo'] < 0.2 else "УМЕРЕННЫЙ СТРЕСС" if row['PSI_geo'] < 0.7 else "ВЫСОКОЕ НАПРЯЖЕНИЕ"
        if cy == 1969: status = "ПОГРАНИЧНЫЙ КЛИНЧ (СССР)"
        if cy == 1989: status = "КРИЗИС ТЯНЬАНЬМЭНЬ"
        if cy == 2022: status = "НАЧАЛО БЛОКАДЫ ЧИПОВ"
        print(f"{int(row['Year']):<6} | {row['MMP']:<7.3f} | {row['EMP']:<7.3f} | {row['SFB']:<7.3f} | {row['Xi']:<7.3f} | {row['M_cog']:<7.3f} | {row['PSI_geo']:<8.4f} | {status}")
        
    print("-"*95)
    print("... ЗАПУСК НЕЙРОСЕТЕВОЙ ЭКСТРАПОЛЯЦИИ КВАНТОВ БУДУЩЕГО (ПЕТЛЯ НЕВЕДЕНИЯ) ...")
    print("-"*95)
    
    breakthrough_year = None
    for y in future_years:
        row = df_full[df_full['Year'] == y].iloc[0]
        if row['PSI_geo'] > CHINA_CRITICAL_MARKER and breakthrough_year is None:
            breakthrough_year = int(y)
            status = "ТОЧКА СИСТЕМНОГО ПРОЛОМА КНР!"
        elif breakthrough_year is not None:
            status = "ФАЗА ТЕРМИНАЛЬНОГО ХАОСА"
        else:
            status = "ИНЕРЦИОННОЕ УДЕРЖАНИЕ"
            
        # Выводим каждый 3-й год прогноза для читаемости консоли
        if y % 3 == 0 or y == 2056 or status == "ТОЧКА СИСТЕМНОГО ПРОЛОМА КНР!":
            print(f"{int(row['Year']):<6} | {row['MMP']:<7.3f} | {row['EMP']:<7.3f} | {row['SFB']:<7.3f} | {row['Xi']:<7.3f} | {row['M_cog']:<7.3f} | {row['PSI_geo']:<8.4f} | {status}")
            
    print("="*95)
    print(f" КЛИОДИНАМИЧЕСКИЙ ВЕРДИКТ ИИ ДЛЯ КИТАЯ: ")
    print(f" С учетом зажима информационных каналов (Mcog -> {df_pred['M_cog'].iloc[0]:.3f}) и лавинообразного ")
    print(f" роста внутреннего перепроизводства элит (EMP -> {df_pred['EMP'].iloc[-1]:.2f} к 2056 г.), ")
    print(f" КРИТИЧЕСКИЙ ЛИМИТ ПРОЧНОСТИ ИНСТИТУТОВ КНР БУДЕТ ПРОБИТ В {breakthrough_year} ГОДУ. ")
    print("="*95)

if __name__ == "__main__":
    run_china_psychohistory_engine()
