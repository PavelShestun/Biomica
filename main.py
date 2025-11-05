# -*- coding: utf-8 -*-
"""
Основной файл для запуска симуляции фаговой устойчивости.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
from scipy.ndimage import convolve

from config import Config

# --- Классы Агентов ---

class Bacterium:
    """Класс для агента-бактерии."""
    def __init__(self, defenses, state='ACTIVE'):
        self.state = state
        self.defenses = defenses
        self.infection_timer = 0
        self.infected_by = None

    def is_vulnerable_to(self, phage):
        return phage.counters.issuperset(self.defenses)

class Phage:
    """Класс для агента-фага."""
    def __init__(self, counters):
        self.counters = counters

# --- Основной класс Симуляции ---

class Simulation:
    """Управляет состоянием и логикой всей симуляции."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.grid1, self.nutrient_grid1 = self._setup_grid(self.cfg.PRODUCER_1_PARAMS)
        self.grid2, self.nutrient_grid2 = self._setup_grid(self.cfg.PRODUCER_2_PARAMS)
        self.history = defaultdict(list)

        self.im_data1 = np.zeros((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE, 4), dtype=np.float32)
        self.im_data2 = np.zeros((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE, 4), dtype=np.float32)

    def _setup_grid(self, params):
        """Создает начальное состояние сеток для одного производителя."""
        grid = np.full((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE), None, dtype=object)
        nutrient_grid = np.full((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE), self.cfg.INITIAL_NUTRIENT_LEVEL, dtype=np.float32)

        y, x = np.ogrid[-self.cfg.CENTER:self.cfg.GRID_SIZE - self.cfg.CENTER,
                        -self.cfg.CENTER:self.cfg.GRID_SIZE - self.cfg.CENTER]
        mask = x*x + y*y > self.cfg.RADIUS*self.cfg.RADIUS
        grid[mask] = -1
        nutrient_grid[mask] = 0

        colony_radius = 5
        for r_off in range(-colony_radius, colony_radius + 1):
            for c_off in range(-colony_radius, colony_radius + 1):
                if r_off**2 + c_off**2 <= colony_radius**2:
                    r, c = self.cfg.CENTER + r_off, self.cfg.CENTER + c_off
                    defenses = params['initial_defenses']()
                    grid[r, c] = Bacterium(defenses=defenses, state='ACTIVE')
        return grid, nutrient_grid

    def _simulation_step(self, grid_of_agents, nutrient_grid, params):
        """Выполняет один шаг симуляции для одной сетки."""
        new_grid = grid_of_agents.copy()
        new_nutrient_grid = nutrient_grid.copy()

        all_coords = [(r, c) for r in range(self.cfg.GRID_SIZE) for c in range(self.cfg.GRID_SIZE)]
        random.shuffle(all_coords)

        for r, c in all_coords:
            agent = grid_of_agents[r, c]
            if agent is None or agent == -1:
                continue

            if isinstance(agent, Bacterium):
                self._update_bacterium(agent, r, c, new_grid, new_nutrient_grid, params)
            elif isinstance(agent, Phage):
                self._update_phage(agent, r, c, new_grid)

        # Диффузия питательных веществ
        new_nutrient_grid = self._diffuse_nutrients(new_nutrient_grid)

        return new_grid, new_nutrient_grid

    def _update_bacterium(self, agent, r, c, new_grid, nutrient_grid, params):
        """Обновляет состояние одной бактерии."""
        if agent.state == 'REFUGE':
            if random.random() < params['p_leave_refuge']: agent.state = 'ACTIVE'
        elif agent.state == 'ACTIVE':
            if random.random() < params['p_enter_refuge']: agent.state = 'REFUGE'

        if agent.state == 'ACTIVE':
            # Рост зависит от локальных нутриентов
            growth_prob = (self.cfg.R_MAX_GROWTH - self.cfg.COST_PER_DEFENSE * len(agent.defenses))
            growth_prob *= nutrient_grid[r, c] # Прямая зависимость от еды

            if random.random() < growth_prob and nutrient_grid[r, c] > self.cfg.NUTRIENT_CONSUMPTION:
                empty_neighbors = [pos for pos in self._get_neighbors(r,c) if new_grid[pos] is None]
                if empty_neighbors:
                    nr, nc = random.choice(empty_neighbors)
                    # Потребление нутриентов
                    nutrient_grid[r, c] -= self.cfg.NUTRIENT_CONSUMPTION
                    new_defenses = self._mutate_repertoire(agent.defenses)
                    new_grid[nr, nc] = Bacterium(defenses=new_defenses, state='ACTIVE')

        elif agent.state == 'INFECTED':
            agent.infection_timer -= 1
            if agent.infection_timer <= 0:
                new_grid[r, c] = None
                phage_parent = agent.infected_by
                burst_size = int(self.cfg.BURST_SIZE_MAX - self.cfg.COST_PER_COUNTER * len(phage_parent.counters))
                empty_neighbors = [pos for pos in self._get_neighbors(r,c, diagonal=True) if new_grid[pos] is None]
                random.shuffle(empty_neighbors)
                for _ in range(burst_size):
                    if not empty_neighbors: break
                    nr, nc = empty_neighbors.pop()
                    new_grid[nr, nc] = Phage(counters=phage_parent.counters)

    def _diffuse_nutrients(self, nutrient_grid):
        """Распространение питательных веществ по среде."""
        kernel = np.array([[0.5, 1, 0.5],
                           [1,  -6,  1],
                           [0.5, 1, 0.5]]) * self.cfg.DIFFUSION_COEFFICIENT

        # Применяем свертку для диффузии
        diffusion = convolve(nutrient_grid, kernel, mode='constant', cval=0.0)

        # Обновляем сетку, не выходя за пределы [0, 1]
        nutrient_grid += diffusion
        np.clip(nutrient_grid, 0, 1, out=nutrient_grid)
        return nutrient_grid


    def _update_phage(self, agent, r, c, new_grid):
        """Обновляет состояние одного фага."""
        if random.random() < self.cfg.P_PHAGE_DECAY:
            new_grid[r, c] = None
            return

        potential_targets = [
            (nr, nc) for nr, nc in self._get_neighbors(r, c)
            if isinstance(new_grid[nr, nc], Bacterium) and new_grid[nr, nc].state == 'ACTIVE' and new_grid[nr, nc].is_vulnerable_to(agent)
        ]

        if potential_targets:
            target_r, target_c = random.choice(potential_targets)
            target_bacterium = new_grid[target_r, target_c]
            if random.random() < self.cfg.P_INFECTION:
                target_bacterium.state = 'INFECTED'
                target_bacterium.infection_timer = self.cfg.LATENCY_PERIOD
                target_bacterium.infected_by = agent
                new_grid[r, c] = None
        else:
            empty_neighbors = [pos for pos in self._get_neighbors(r, c) if new_grid[pos] is None]
            if empty_neighbors:
                nr, nc = random.choice(empty_neighbors)
                new_grid[nr, nc], new_grid[r, c] = new_grid[r, c], None

    def _add_phages(self, grid):
        """Добавляет фагов в сетку."""
        empty_coords = np.argwhere(grid == None)
        if len(empty_coords) > self.cfg.INITIAL_PHAGE_COUNT:
            indices = np.random.choice(len(empty_coords), self.cfg.INITIAL_PHAGE_COUNT, replace=False)
            for idx in indices:
                r, c = empty_coords[idx]
                grid[r, c] = Phage(counters=self.cfg.PHAGE_ATTACK_REPERTOIRE)

    def _get_neighbors(self, r, c, diagonal=False):
        """Вспомогательная функция для получения координат соседей."""
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        neighbors = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.cfg.GRID_SIZE and 0 <= nc < self.cfg.GRID_SIZE:
                neighbors.append((nr, nc))
        return neighbors

    def _mutate_repertoire(self, repertoire):
        """Мутация репертуара защит."""
        new_repertoire = repertoire.copy()
        if random.random() < self.cfg.P_MUTATION:
            if random.random() < 0.5 and len(new_repertoire) > 0:
                new_repertoire.remove(random.choice(list(new_repertoire)))
            elif len(new_repertoire) < self.cfg.N_DEFENSE_TYPES:
                possible = list(set(range(self.cfg.N_DEFENSE_TYPES)) - new_repertoire)
                if possible:
                    new_repertoire.add(random.choice(possible))
        return new_repertoire

    def run_animation(self):
        """Настраивает и запускает Matplotlib анимацию."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

        im1 = axes[0].imshow(self.im_data1, interpolation='none')
        axes[0].set_title(self.cfg.PRODUCER_1_PARAMS['name'])
        axes[0].set_xticks([]); axes[0].set_yticks([])

        im2 = axes[1].imshow(self.im_data2, interpolation='none')
        axes[1].set_title(self.cfg.PRODUCER_2_PARAMS['name'])
        axes[1].set_xticks([]); axes[1].set_yticks([])

        ax_plot = axes[2]
        lines = {
            'b1': ax_plot.plot([], [], color='#4285F4', label='Бактерии 1')[0],
            'p1': ax_plot.plot([], [], color='#A5C8FF', linestyle=':', label='Фаги 1')[0],
            'b2': ax_plot.plot([], [], color='#EA4335', label='Бактерии 2')[0],
            'p2': ax_plot.plot([], [], color='#FADADD', linestyle=':', label='Фаги 2')[0]
        }

        ax_plot.set_title("Динамика популяций")
        ax_plot.set_xlabel("Время (шаги симуляции)")
        ax_plot.set_ylabel("Численность (log шкала)")
        ax_plot.set_xlim(0, self.cfg.N_FRAMES)
        ax_plot.set_yscale('log')
        ax_plot.set_ylim(1, np.sum(self.grid1 != -1) * 1.5)
        ax_plot.axvline(self.cfg.FRAME_ADD_PHAGE, color='yellow', linestyle='--', lw=1, label='Внесение фагов')
        ax_plot.legend(loc='lower left', fontsize='small')
        ax_plot.grid(True, alpha=0.3)
        plt.tight_layout()

        def update(frame):
            """Основная функция обновления для анимации."""
            if frame == self.cfg.FRAME_ADD_PHAGE:
                self._add_phages(self.grid1)
                self._add_phages(self.grid2)

            self.grid1, self.nutrient_grid1 = self._simulation_step(self.grid1, self.nutrient_grid1, self.cfg.PRODUCER_1_PARAMS)
            self.grid2, self.nutrient_grid2 = self._simulation_step(self.grid2, self.nutrient_grid2, self.cfg.PRODUCER_2_PARAMS)

            self._update_im_data(self.grid1, self.im_data1, self.nutrient_grid1)
            self._update_im_data(self.grid2, self.im_data2, self.nutrient_grid2)
            im1.set_data(self.im_data1)
            im2.set_data(self.im_data2)

            self.history['t'].append(frame)
            self._collect_history()

            for key, line in lines.items():
                line.set_data(self.history['t'], self.history[key])

            if frame % 20 == 0:
                print(f"Кадр {frame}/{self.cfg.N_FRAMES}")

            return list(lines.values()) + [im1, im2]

        anim = FuncAnimation(fig, update, frames=self.cfg.N_FRAMES, interval=80, blit=True)
        plt.show()

    def _update_im_data(self, grid, im_data, nutrient_grid):
        """Обновляет RGBA-массив для отрисовки."""
        # Фон теперь зависит от уровня питательных веществ
        background_color = np.clip(nutrient_grid * 0.2, 0, 1)[:, :, np.newaxis]
        im_data[:, :, :3] = background_color
        im_data[:, :, 3] = 1.0  # Alpha channel

        colors = {
            'ACTIVE': [0.2, 0.8, 0.2, 1.0], 'REFUGE': [0.1, 0.3, 0.1, 1.0],
            'INFECTED': [1.0, 1.0, 0.0, 1.0], 'PHAGE': [1.0, 0.2, 0.2, 1.0]
        }
        for r in range(self.cfg.GRID_SIZE):
            for c in range(self.cfg.GRID_SIZE):
                agent = grid[r, c]
                if isinstance(agent, Bacterium): im_data[r, c] = colors[agent.state]
                elif isinstance(agent, Phage): im_data[r, c] = colors['PHAGE']
                elif agent == -1: im_data[r,c] = [0.1, 0.1, 0.1, 1.0]

    def _collect_history(self):
        """Собирает статистику по популяциям для графика."""
        counts = {'b1': 0, 'p1': 0, 'b2': 0, 'p2': 0}
        for agent in self.grid1.ravel():
            if isinstance(agent, Bacterium): counts['b1'] += 1
            elif isinstance(agent, Phage): counts['p1'] += 1
        for agent in self.grid2.ravel():
            if isinstance(agent, Bacterium): counts['b2'] += 1
            elif isinstance(agent, Phage): counts['p2'] += 1

        for key, count in counts.items():
            self.history[key].append(count if count > 0 else 1)

    def run_headless(self):
        """
        Запускает симуляцию без визуализации для сбора данных.
        """
        for frame in range(self.cfg.N_FRAMES):
            if frame == self.cfg.FRAME_ADD_PHAGE:
                self._add_phages(self.grid1)
                self._add_phages(self.grid2)

            self.grid1, self.nutrient_grid1 = self._simulation_step(self.grid1, self.nutrient_grid1, self.cfg.PRODUCER_1_PARAMS)
            self.grid2, self.nutrient_grid2 = self._simulation_step(self.grid2, self.nutrient_grid2, self.cfg.PRODUCER_2_PARAMS)

            if frame % 100 == 0: # Выводим прогресс
                print(f"  Шаг {frame}/{self.cfg.N_FRAMES}")

        # Собираем итоговые данные
        final_counts = {'b1': 0, 'p1': 0, 'b2': 0, 'p2': 0}
        for agent in self.grid1.ravel():
            if isinstance(agent, Bacterium): final_counts['b1'] += 1
            elif isinstance(agent, Phage): final_counts['p1'] += 1
        for agent in self.grid2.ravel():
            if isinstance(agent, Bacterium): final_counts['b2'] += 1
            elif isinstance(agent, Phage): final_counts['p2'] += 1
        return final_counts

if __name__ == "__main__":
    # Этот блок теперь используется только для визуального запуска одной симуляции
    config = Config()
    sim = Simulation(config)
    sim.run_animation()
