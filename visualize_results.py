# -*- coding: utf-8 -*-
"""
Скрипт для визуализации результатов серии экспериментов.

Этот скрипт загружает данные из CSV-файла, сгенерированного
`run_experiments.py`, и строит ящичковые диаграммы (box plots)
для анализа влияния различных параметров на выживаемость культур.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_varied_parameter(df):
    """
    Находит параметр, значения которого менялись в ходе эксперимента.
    Предполагается, что такой параметр только один.
    """
    # Исключаем служебные колонки
    cols_to_exclude = ['combo_id', 'run_id', 'final_b1', 'final_p1', 'final_b2', 'final_p2']
    param_cols = [col for col in df.columns if col not in cols_to_exclude]

    for col in param_cols:
        if df[col].nunique() > 1:
            return col
    return None

def plot_experiment_results(df, varied_param):
    """
    Строит box plot для визуализации влияния одного параметра на
    конечную численность бактерий для обоих производителей.
    """
    if varied_param is None:
        print("Не найдено ни одного варьируемого параметра. Построение графика невозможно.")
        # Если ничего не варьировалось, строим простой график как раньше
        plot_data = pd.melt(df, value_vars=['final_b1', 'final_b2'],
                            var_name='Producer', value_name='Final Population')
        varied_param = 'Producer' # Используем Producer как ось X
    else:
        # Подготавливаем данные для плоттинга
        plot_data = pd.melt(df, id_vars=[varied_param],
                            value_vars=['final_b1', 'final_b2'],
                            var_name='Producer', value_name='Final Population')

    plot_data['Producer'] = plot_data['Producer'].map({
        'final_b1': 'Производитель 1 (Устойчивый)',
        'final_b2': 'Производитель 2 (Уязвимый)'
    })

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=varied_param, y='Final Population', hue='Producer', data=plot_data,
                palette={'Производитель 1 (Устойчивый)': '#4285F4', 'Производитель 2 (Уязвимый)': '#EA4335'})

    plt.title(f'Влияние параметра "{varied_param}" на выживаемость культур', fontsize=16, pad=20)
    plt.xlabel(f'Значение параметра: {varied_param}', fontsize=12)
    plt.ylabel('Конечная численность бактерий', fontsize=12)
    plt.yscale('log') # Логарифмическая шкала часто бывает полезна для популяций
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title='Тип культуры')

    # Сохраняем график в файл
    output_path = f'experiment_analysis_{varied_param}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График анализа сохранен в: {output_path}")
    plt.show()

def main(filepath='experiment_results.csv'):
    """
    Основная функция для загрузки и анализа данных.
    """
    try:
        results_df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filepath}' не найден.")
        print("Пожалуйста, сначала запустите 'run_experiments.py' для генерации данных.")
        return

    print("--- Анализ результатов серии экспериментов ---")
    print(f"Загружено {len(results_df)} записей из '{filepath}'.\n")

    # Автоматически находим параметр, который варьировался
    varied_param = find_varied_parameter(results_df)

    if varied_param:
        print(f"Обнаружен варьируемый параметр: '{varied_param}'")
        # Показываем агрегированную статистику
        summary = results_df.groupby([varied_param])[['final_b1', 'final_b2']].describe()
        print("\nАгрегированная статистика:")
        print(summary)
    else:
         print("Все параметры были фиксированы в данном эксперименте.")
         print("\nОписательная статистика:")
         print(results_df[['final_b1', 'final_b2']].describe())


    # Визуализация результатов
    plot_experiment_results(results_df, varied_param)


if __name__ == "__main__":
    main()
