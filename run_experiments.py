# -*- coding: utf-8 -*-
"""
Усовершенствованный скрипт для проведения вычислительных экспериментов.

Этот скрипт позволяет систематически исследовать влияние различных параметров
на исход симуляции. Он итерирует по сетке заданных параметров, запускает
симуляцию многократно для каждого набора и сохраняет результаты.
Использует tqdm для наглядного отображения прогресса.
"""
import pandas as pd
from main import Simulation
from config import Config
import time
from tqdm import tqdm
import itertools

# --- Сетка параметров для исследования ---
# Каждый элемент в списке 'values' будет использован в отдельном эксперименте
PARAM_GRID = {
    'grid_size': {'values': [100]},
    'n_steps': {'values': [1000]},
    'initial_bacteria_ratio': {'values': [0.1]},
    'phage_burst_size': {'values': [5]},
    # Пример: исследование влияния вероятности ухода в убежище
    'p1_enter_refuge': {'values': [0.01, 0.05, 0.1]},
    'p2_enter_refuge': {'values': [0.001]} # Фиксируем для второго производителя
}

# --- Общие параметры эксперимента ---
N_RUNS_PER_CONFIG = 5 # Количество повторов для каждой комбинации параметров

def run_experiment_grid():
    """
    Выполняет симуляции по всей сетке параметров (PARAM_GRID).
    """
    # Извлекаем имена и списки значений параметров
    param_names = list(PARAM_GRID.keys())
    param_value_lists = [v['values'] for v in PARAM_GRID.values()]

    # Создаем все возможные комбинации параметров
    param_combinations = list(itertools.product(*param_value_lists))

    total_experiments = len(param_combinations) * N_RUNS_PER_CONFIG
    print(f"--- Начало серии экспериментов ---")
    print(f"Всего будет проведено симуляций: {total_experiments}")
    print(f"Количество комбинаций параметров: {len(param_combinations)}")
    print(f"Повторов на комбинацию: {N_RUNS_PER_CONFIG}")

    all_results = []

    # Основной цикл по комбинациям параметров с использованием tqdm
    with tqdm(total=total_experiments, desc="Общий прогресс") as pbar:
        for combo_idx, param_combo in enumerate(param_combinations):

            # Создаем словарь текущей конфигурации
            current_params = dict(zip(param_names, param_combo))

            # Запускаем N_RUNS_PER_CONFIG симуляций для данной конфигурации
            for run_idx in range(N_RUNS_PER_CONFIG):

                # Создаем экземпляр конфига и переопределяем параметры
                config = Config()
                for param, value in current_params.items():
                    setattr(config, param, value)

                sim = Simulation(config)
                final_counts = sim.run_headless()

                # Собираем строку результата
                result_row = {
                    'combo_id': combo_idx,
                    'run_id': run_idx,
                    **current_params, # добавляем параметры эксперимента
                    'final_b1': final_counts['b1'],
                    'final_p1': final_counts['p1'],
                    'final_b2': final_counts['b2'],
                    'final_p2': final_counts['p2']
                }
                all_results.append(result_row)
                pbar.update(1)

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    start_time = time.time()

    # Запускаем серию экспериментов
    results_df = run_experiment_grid()

    output_path = 'experiment_results.csv'
    results_df.to_csv(output_path, index=False)

    end_time = time.time()

    print(f"\n--- Серия экспериментов завершена за {end_time - start_time:.2f} секунд ---")
    print(f"Результаты сохранены в файл: {output_path}")
    print(f"\nИтоговая таблица результатов (первые 5 строк):")
    print(results_df.head())
