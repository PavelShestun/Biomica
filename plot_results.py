# -*- coding: utf-8 -*-
"""
Скрипт для анализа и визуализации результатов экспериментов.

Этот скрипт загружает данные из CSV-файла, сгенерированного
run_experiments.py, вычисляет средние значения и строит
сравнительные графики для анализа выживаемости культур.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_survival_distribution(df):
    """
    Строит violin plot для визуализации распределения конечной
    численности бактерий для обоих производителей.
    """
    # Подготавливаем данные для плоттинга
    plot_data = pd.melt(df, value_vars=['b1', 'b2'],
                        var_name='Producer', value_name='Final Population')

    plot_data['Producer'] = plot_data['Producer'].map({
        'b1': 'Производитель 1 (Устойчивый)',
        'b2': 'Производитель 2 (Уязвимый)'
    })

    plt.figure(figsize=(10, 7))
    sns.violinplot(x='Producer', y='Final Population', data=plot_data, palette=['#4285F4', '#EA4335'])

    # Добавляем точки для каждого отдельного прогона
    sns.stripplot(x='Producer', y='Final Population', data=plot_data, color='k', alpha=0.5, jitter=0.2)

    plt.title(f'Распределение выживаемости культур после {len(df)} прогонов', fontsize=16)
    plt.xlabel('Тип культуры', fontsize=12)
    plt.ylabel('Конечная численность бактерий (log шкала)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Сохраняем график в файл
    output_path = 'survival_distribution.png'
    plt.savefig(output_path, dpi=300)
    print(f"График распределения выживаемости сохранен в: {output_path}")
    plt.show()

def main(filepath='results.csv'):
    """
    Основная функция для загрузки и анализа данных.
    """
    try:
        results_df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filepath}' не найден.")
        print("Пожалуйста, сначала запустите 'run_experiments.py' для генерации данных.")
        return

    print("--- Анализ результатов эксперимента ---")
    print(f"Загружено {len(results_df)} записей из '{filepath}'.\n")

    # Вывод описательной статистики
    print("Описательная статистика по конечной численности:")
    stats = results_df[['b1', 'b2']].describe().rename(columns={
        'b1': 'Производитель 1',
        'b2': 'Производитель 2'
    })
    print(stats)

    # Визуализация результатов
    plot_survival_distribution(results_df)


if __name__ == "__main__":
    main()
