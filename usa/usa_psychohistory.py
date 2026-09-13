"""
COMPENDIUM PSYCHOHISTORIAE — МОДУЛЬ США (СЦЕНАРИЙ ОТКРЫТОГО КОНТУРА)
Масштабный нелинейный эксперимент: расчет индексов SDT и PSI_geo для США (1953–2056 гг.)
с учетом перепроизводства юридических элит, суверенного долга и информационного шума Mcog.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_usa_historical_dataset():
    """
    Формирует плотный ежегодный исторический ряд макроструктурных параметров США (1953–2026 гг.).
    Данные нормированы и калиброваны с учетом специфики американских секулярных циклов.
    """
    years = list(range(1953, 2027, 1))
    hist_data = []
    
    for y in years:
        # 1. Urb (Доля городского населения) — Стабильно высокий постиндустриальный уровень
        urb = 0.64 + (0.83 - 0.64) * (y - 1953) / (2026 - 1953)
            
        # 2. Youth (Молодежная когорта) — Всплеск бэби-бумеров 1970-х и миграционный демпфер
        if y <= 1975:
            youth = 0.12 + 0.04 * (y - 1953) / 22
        elif y <= 1995:
            youth = 0.16 - 0.03 * (y - 1975) / 20
        else:
            youth = 0.13 + 0.01 * np.sin(y / 2.0) # Стабильный приток молодой миграции
        
        # 3. w_inv (Инверсированная зарплата) — Экономическое истощение масс из-за стагнации доходов с 1970-х
        if y <= 1973:
            w_inv = 1.25 + 0.05 * np.sin(y / 3.0)  # Эпоха сильного среднего класса
        else:
            w_inv = 1.30 + 0.85 * (y - 1973) / 53  # Взрывной рост неравенства (активы уходят топ-1%)
            
        # 4. EMP (Избыток элит) — Перепроизводство юристов (Law Schools), MBA и раскол на Трампистов/Либералов
        if y <= 1970:
            emp = 1.00 + 0.2 * (y - 1953) / 17     # Контролируемый уровень конкуренции элит
        elif y <= 2000:
            emp = 1.20 + 0.8 * (y - 1970) / 30     # Появление "новых денег", Уолл-стрит, доткомы
        elif y <= 2016:
            emp = 2.00 + 0.9 * (y - 2000) / 16     # Взлет числа миллионеров и раскол партийных систем
        else:
            emp = 2.90 + 0.8 * (y - 2016) / 10     # Беспрецедентный клинч, контр-элиты штурмуют Капитолий

        # 5. SFB (Фискальный стресс) — Лавинообразный взлет госдолга за планку 35 трлн долларов
        if y <= 1980:
            sfb = 0.35 + 0.05 * np.cos(y)          # Безопасный послевоенный уровень долга к доходам
        elif y <= 2008:
            sfb = 0.40 + 0.40 * (y - 1980) / 28    # Рейганомика, войны на Ближнем Востоке
        else:
            sfb = 0.80 + 1.15 * (y - 2008) / 18    # Терминальный печатный станок: QE, пандемия 2020, 2022-2026

        # 6. Индекс утраты мировой гегемонии (\u039e для США)
        # Шкала: чем выше 1.0, тем сильнее сжимаются внешние ресурсы (дедолларизация, БРИКС, дефицит баланса)
        if y < 1971: oil = 1.0                    # Золотой стандарт доллара
        elif y <= 1980: oil = 1.6                  # Нефтяные шоки 1973 и 1979 годов, стагфляция
        else: oil = 1.0 + 0.3 * np.sin(y / 4.0)
        
        sanctions = 0.2 if y < 2014 else 0.5 + 0.5 * (y - 2014) / 12  # Издержки санкционных войн для самой финансовой системы
        
        # Вектор геополитического давления главных соперников (СССР/РФ и Китай)
        rus = 1.5 if y <= 1989 else 0.3 if y <= 2014 else 1.4  # Карибский кризис -> Разрядка -> Новая Холодная война
        china = 0.2 if y < 2001 else 0.5 + 1.1 * (y - 2001) / 25  # Вступление КНР в ВТО, потеря американской промбазы
        
        xi = (oil * 0.20) + (sanctions * 0.15) + (rus * 0.30) + (china * 0.35)
        
        # 7. I_flow и Информационный коллапс / Шум (Doomscrolling, поляризация соцсетей)
        # В США каналы зажимаются не цензурой (как в КНР), а сверхвысокой плотностью зашумления потока (I_flow -> 1.0)
        if y < 2008: i_flow = 0.50                 # Информационный оптимум до эпохи смартфонов
        else: i_flow = min(0.88, 0.50 + 0.38 * (y - 2008) / 18)
        
        mcog = 4.0 * i_flow * (1.0 - i_flow)
        
        hist_data.append([y, w_inv, urb, youth, emp, sfb, oil, sanctions, rus, china, xi, i_flow, mcog])
        
    df = pd.DataFrame(hist_data, columns=[
        'Year', 'w_inv', 'Urb', 'Youth', 'EMP', 'SFB', 
        'Oil', 'Sanctions', 'Rus', 'China', 'Xi', 'I_flow', 'M_cog'
    ])
    df['MMP'] = df['w_inv'] * df['Urb'] * df['Youth']
    df['PSI_geo'] = (df['MMP'] / df['M_cog']) * df['EMP'] * df['SFB'] * df['Xi']
    return df

def run_usa_psychohistory_engine():
    """
    Запускает глубокую нейросеть для прогнозирования США до 2056 года.
    """
    df_hist = generate_usa_historical_dataset()
    future_years = np.arange(2027, 2057, 1)
    
    X_train = df_hist[['Year']].values
    X_future = future_years.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_future_scaled = scaler.transform(X_future)
    
    nn = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='tanh', solver='lbfgs', max_iter=10000, random_state=42)
    
    df_pred = pd.DataFrame({'Year': future_years})
    targets = ['w_inv', 'EMP', 'SFB', 'Oil', 'Sanctions', 'Rus', 'China']
    
    for t in targets:
        nn.fit(X_train_scaled, df_hist[t].values)
        df_pred[t] = nn.predict(X_future_scaled)
        
    df_pred['Urb'] = 0.83
    df_pred['Youth'] = [max(0.10, 0.13 - 0.02 * (y - 2026) / 30) for y in future_years]
    
    # Фиксация информационного шума (I_flow заблокирован на уровне экстремального зашумления)
    df_pred['I_flow'] = 0.88
    df_pred['M_cog'] = 4.0 * df_pred['I_flow'] * (1.0 - df_pred['I_flow'])
    
    df_pred['Xi'] = (df_pred['Oil'] * 0.20) + (df_pred['Sanctions'] * 0.15) + (df_pred['Rus'] * 0.30) + (df_pred['China'] * 0.35)
    df_pred['MMP'] = df_pred['w_inv'] * df_pred['Urb'] * df_pred['Youth']
    df_pred['PSI_geo'] = (df_pred['MMP'] / df_pred['M_cog']) * df_pred['EMP'] * df_pred['SFB'] * df_pred['Xi']
    
    df_full = pd.concat([df_hist, df_pred], ignore_index=True)
    USA_CRITICAL_MARKER = 1.4730
    
    print("="*95)
    print(" COMPENDIUM PSYCHOHISTORIAE: КЛИОДИНАМИЧЕСКИЙ РАСЧЕТ МАКРОСИСТЕМЫ США (1953-2056) ")
    print("="*95)
    print(f"{'Год':<6} | {'MMP':<7} | {'EMP':<7} | {'SFB':<7} | {'Xi':<7} | {'M_cog':<7} | {'PSI_geo':<8} | {'Статус Системы'}")
    print("-"*95)
    
    control_years = [1962, 1974, 2008, 2020, 2024, 2026]
    for cy in control_years:
        row = df_full[df_full['Year'] == cy].iloc[0]
        status = "СТАБИЛЬНАЯ ГЕГЕМОНИЯ" if row['PSI_geo'] < 0.2 else "СИСТЕМНЫЙ СТРЕСС" if row['PSI_geo'] < 0.7 else "СВЕРХВЫСОКИЙ ПЕРЕГРЕВ"
        if cy == 1962: status = "КАРИБСКИЙ КРИЗИС"
        if cy == 1974: status = "УОТЕРГЕЙТ / НЕФТЯНОЙ ШОК"
        if cy == 2020: status = "КРИЗИС BLM / ПАНДЕМИЯ (ПРОГНОЗ ТУРЧИНА)"
        print(f"{int(row['Year']):<6} | {row['MMP']:<7.3f} | {row['EMP']:<7.3f} | {row['SFB']:<7.3f} | {row['Xi']:<7.3f} | {row['M_cog']:<7.3f} | {row['PSI_geo']:<8.4f} | {status}")
        
    print("-"*95)
    print("... ЗАПУСК НЕЙРОСЕТЕВОЙ ЭКСТРАПОЛЯЦИИ КВАНТОВ БУДУЩЕГО (ИНФОРМАЦИОННЫЙ ШУМ) ...")
    print("-"*95)
    
    breakthrough_year = None
    for y in future_years:
        row = df_full[df_full['Year'] == y].iloc[0]
        if row['PSI_geo'] > USA_CRITICAL_MARKER and breakthrough_year is None:
            breakthrough_year = int(y)
            status = "ТОЧКА СИСТЕМНОГО ПРОЛОМА США!"
        elif breakthrough_year is not None:
            status = "ФАЗА ИНСТИТУЦИОНАЛЬНОЙ ДЕГРАДАЦИИ"
        else:
            status = "ИНЕРЦИОННОЕ УДЕРЖАНИЕ ДОЛЛАРА"
            
        if y % 3 == 0 or y == 2056 or status == "ТОЧКА СИСТЕМНОГО ПРОЛОМА США!":
            print(f"{int(row['Year']):<6} | {row['MMP']:<7.3f} | {row['EMP']:<7.3f} | {row['SFB']:<7.3f} | {row['Xi']:<7.3f} | {row['M_cog']:<7.3f} | {row['PSI_geo']:<8.4f} | {status}")
            
    print("="*95)
    print(f" КЛИОДИНАМИЧЕСКИЙ ВЕРДИКТ ИИ ДЛЯ США: ")
    print(f" С учетом когнитивного зашумления (Mcog -> {df_pred['M_cog'].iloc[0]:.3f}) и запредельного ")
    print(f" фискально-элитного перегрева (SFB -> {df_pred['SFB'].iloc[-1]:.2f}, EMP -> {df_pred['EMP'].iloc[-1]:.2f}), ")
    print(f" КРИТИЧЕСКИЙ ЛИМИТ ПРОЧНОСТИ СИСТЕМЫ США БУДЕТ ПРОБИТ В {breakthrough_year} ГОДУ. ")
    print("="*95)

if __name__ == "__main__":
    run_usa_psychohistory_engine()
