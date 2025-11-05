# -*- coding: utf-8 -*-
"""
Скрипт для проведения вычислительных экспериментов.

Этот скрипт запускает симуляцию в "безголовом" режиме (без анимации)
многократно для сбора статистически значимых данных. Результаты
сохраняются в CSV файл для дальнейшего анализа.
"""
import pandas as pd
from main import Simulation
from config import Config
import time

# --- Параметры эксперимента ---
N_RUNS = 2  # Количество повторных прогонов для каждого набора параметров

def run_single_experiment():
    """
    Выполняет N_RUNS симуляций и возвращает результаты в виде списка словарей.
    """
    results_list = []

    for i in range(N_RUNS):
        print(f"Запуск симуляции №{i+1}/{N_RUNS}...")
        start_time = time.time()

        # Создаем новый экземпляр конфигурации и симуляции для каждого прогона
        # чтобы гарантировать независимость результатов
        config = Config()
        sim = Simulation(config)

        # Запускаем симуляцию в режиме без визуализации
        final_counts = sim.run_headless()

        # Добавляем номер прогона в результаты
        result_row = final_counts
        result_row['run_id'] = i

        results_list.append(result_row)

        end_time = time.time()
        print(f"  Завершено за {end_time - start_time:.2f} секунд. Выжило (Б1/Б2): {final_counts['b1']}/{final_counts['b2']}")

    return results_list

if __name__ == "__main__":
    print("--- Начало вычислительного эксперимента ---")

    # Запускаем серию симуляций
    results = run_single_experiment()

    # Преобразуем результаты в DataFrame и сохраняем в CSV
    results_df = pd.DataFrame(results)

    # Указываем порядок столбцов для удобства
    column_order = ['run_id', 'b1', 'p1', 'b2', 'p2']
    results_df = results_df[column_order]

    output_path = 'results.csv'
    results_df.to_csv(output_path, index=False)

    print(f"\n--- Эксперимент завершен ---")
    print(f"Результаты сохранены в файл: {output_path}")

    # Выводим базовую статистику по результатам
    print("\nКраткая статистика по результатам:")
    print(results_df.describe())
