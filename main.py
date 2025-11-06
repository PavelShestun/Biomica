# -*- coding: utf-8 -*-
"""
Основной файл для запуска симуляции фаговой устойчивости.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
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
    def __init__(self, counters, is_mutant=False):
        self.counters = counters
        self.is_mutant = is_mutant


# --- Основной класс Симуляции ---
class PetriDish:
    """Класс для управления состоянием одной чашки Петри."""
    def __init__(self, grid_size, params, cfg):
        self.grid_size = grid_size
        self.center = grid_size // 2
        self.params = params
        self.cfg = cfg
        self.grid = np.full((grid_size, grid_size), None, dtype=object)
        self.bacteria = {}  # {(r, c): Bacterium}
        self.phages = {}    # {(r, c): Phage}

        # Инициализация среды
        y, x = np.ogrid[-self.center:self.grid_size - self.center, -self.center:self.grid_size - self.center]
        mask = x*x + y*y > self.cfg.RADIUS*self.cfg.RADIUS
        self.grid[mask] = -1 # -1 означает "стена"

    def add_agent(self, r, c, agent):
        """Добавляет агента в указанную клетку."""
        self.grid[r, c] = agent
        if isinstance(agent, Bacterium):
            self.bacteria[(r, c)] = agent
        elif isinstance(agent, Phage):
            self.phages[(r, c)] = agent

    def remove_agent(self, r, c):
        """Удаляет агента из указанной клетки."""
        agent = self.grid[r, c]
        self.grid[r, c] = None
        if isinstance(agent, Bacterium):
            self.bacteria.pop((r, c), None)
        elif isinstance(agent, Phage):
            self.phages.pop((r, c), None)
        return agent

    def move_agent(self, r_old, c_old, r_new, c_new):
        """Перемещает агента из одной клетки в другую."""
        agent = self.remove_agent(r_old, c_old)
        if agent:
            self.add_agent(r_new, c_new, agent)

class Simulation:
    """Управляет состоянием и логикой всей симуляции."""
    def __init__(self, cfg, use_headless=False):
        self.cfg = cfg
        self.use_headless = use_headless
        self.frame = 0
        self.petri_dish_a = None
        self.petri_dish_b = None

        if not self.use_headless:
            self.history = defaultdict(list)
            self.im_data_a = np.zeros((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE, 4), dtype=np.float32)
            self.im_data_b = np.zeros((self.cfg.GRID_SIZE, self.cfg.GRID_SIZE, 4), dtype=np.float32)

    def initialize_petri_dishes(self, init_a=True, init_b=True):
        """Инициализирует чашки Петри."""
        if init_a:
            self.petri_dish_a = self._setup_petri_dish(self.cfg.PRODUCER_1_PARAMS)
        if init_b:
            self.petri_dish_b = self._setup_petri_dish(self.cfg.PRODUCER_2_PARAMS)

    def _setup_petri_dish(self, params):
        """Создает и заселяет одну чашку Петри."""
        dish = PetriDish(self.cfg.GRID_SIZE, params, self.cfg)

        # Размещаем начальную колонию бактерий
        coords_to_populate = []
        for r_offset in range(-5, 6):
            for c_offset in range(-5, 6):
                if r_offset**2 + c_offset**2 <= 25:
                     coords_to_populate.append((dish.center + r_offset, dish.center + c_offset))

        # Убедимся, что не пытаемся разместить больше бактерий, чем есть места
        num_to_place = min(self.cfg.N_BACTERIA, len(coords_to_populate))
        selected_coords = random.sample(coords_to_populate, num_to_place)

        for r, c in selected_coords:
            defenses = params['initial_defenses']()
            bacterium = Bacterium(defenses=defenses, state='ACTIVE')
            dish.add_agent(r, c, bacterium)
        return dish

    def step(self):
        """Выполняет один шаг симуляции для всех активных чашек."""
        # События по расписанию
        if self.frame == self.cfg.FRAME_ADD_PHAGE:
            if self.petri_dish_a: self._add_phages(self.petri_dish_a)
            if self.petri_dish_b: self._add_phages(self.petri_dish_b)

        if self.frame == self.cfg.FRAME_MUTANT_PHAGE_APPEARS:
            if self.petri_dish_a: self._add_mutant_phages(self.petri_dish_a)
            if self.petri_dish_b: self._add_mutant_phages(self.petri_dish_b)

        # Обновление состояния
        if self.petri_dish_a: self._simulation_step_for_dish(self.petri_dish_a)
        if self.petri_dish_b: self._simulation_step_for_dish(self.petri_dish_b)

        self.frame += 1

    def _simulation_step_for_dish(self, dish):
        """Выполняет один шаг симуляции для одной чашки Петри."""
        # Обновляем бактерии
        for (r, c), bacterium in list(dish.bacteria.items()):
            self._update_bacterium(bacterium, r, c, dish)

        # Обновляем фаги
        for (r, c), phage in list(dish.phages.items()):
            self._update_phage(phage, r, c, dish)

    def _update_bacterium(self, bacterium, r, c, dish):
        """Обновляет состояние одной бактерии."""
        state_changed = False
        # Переходы состояний
        if bacterium.state == 'REFUGE':
            if random.random() < dish.params['p_leave_refuge']:
                bacterium.state = 'ACTIVE'
                state_changed = True
        elif bacterium.state == 'ACTIVE':
            if random.random() < dish.params['p_enter_refuge']:
                bacterium.state = 'REFUGE'
                state_changed = True

        # Рост и размножение
        if bacterium.state == 'ACTIVE':
            defense_cost = sum(self.cfg.DEFENSE_COSTS.get(d, 0) for d in bacterium.defenses)
            growth_prob = self.cfg.R_MAX_GROWTH - defense_cost
            if state_changed:
                growth_prob -= self.cfg.COST_PER_REFUGE_TRANSITION

            if random.random() < growth_prob:
                empty_neighbors = [pos for pos in self._get_neighbors(r, c) if dish.grid[pos] is None]
                if empty_neighbors:
                    nr, nc = random.choice(empty_neighbors)
                    new_defenses = self._mutate_repertoire(bacterium.defenses)
                    new_bacterium = Bacterium(defenses=new_defenses, state='ACTIVE')
                    dish.add_agent(nr, nc, new_bacterium)

        # Обработка инфекции
        elif bacterium.state == 'INFECTED':
            bacterium.infection_timer -= 1
            if bacterium.infection_timer <= 0:
                dish.remove_agent(r, c) # Бактерия лизируется
                phage_parent = bacterium.infected_by
                burst_size = int(self.cfg.BURST_SIZE_MAX - self.cfg.COST_PER_COUNTER * len(phage_parent.counters))
                empty_neighbors = [pos for pos in self._get_neighbors(r, c, diagonal=True) if dish.grid[pos] is None]
                random.shuffle(empty_neighbors)
                for _ in range(burst_size):
                    if not empty_neighbors: break
                    nr, nc = empty_neighbors.pop()
                    new_phage = Phage(counters=phage_parent.counters, is_mutant=phage_parent.is_mutant)
                    dish.add_agent(nr, nc, new_phage)

    def _update_phage(self, phage, r, c, dish):
        """Обновляет состояние одного фага."""
        if random.random() < self.cfg.P_PHAGE_DECAY:
            dish.remove_agent(r, c)
            return

        # Поиск и заражение цели
        potential_targets = []
        for nr, nc in self._get_neighbors(r, c):
            agent = dish.grid[nr, nc]
            if isinstance(agent, Bacterium) and agent.state == 'ACTIVE' and agent.is_vulnerable_to(phage):
                potential_targets.append((nr, nc))

        if potential_targets:
            target_r, target_c = random.choice(potential_targets)
            if random.random() < self.cfg.P_INFECTION:
                target_bacterium = dish.grid[target_r, target_c]
                target_bacterium.state = 'INFECTED'
                target_bacterium.infection_timer = self.cfg.LATENCY_PERIOD
                target_bacterium.infected_by = phage
                dish.remove_agent(r, c) # Фаг исчезает
                return

        # Перемещение
        empty_neighbors = [pos for pos in self._get_neighbors(r, c) if dish.grid[pos] is None]
        if empty_neighbors:
            nr, nc = random.choice(empty_neighbors)
            dish.move_agent(r, c, nr, nc)

    def _add_phages(self, dish, is_mutant=False, repertoire=None):
        """Добавляет фагов в чашку Петри."""
        if repertoire is None:
            repertoire = self.cfg.PHAGE_ATTACK_REPERTOIRE if not is_mutant else self.cfg.MUTANT_PHAGE_REPERTOIRE

        empty_coords = np.argwhere(dish.grid == None)
        if len(empty_coords) > self.cfg.INITIAL_PHAGE_COUNT:
            indices = np.random.choice(len(empty_coords), self.cfg.INITIAL_PHAGE_COUNT, replace=False)
            for idx in indices:
                r, c = empty_coords[idx]
                new_phage = Phage(counters=repertoire, is_mutant=is_mutant)
                dish.add_agent(r, c, new_phage)

    def _add_mutant_phages(self, dish):
        """Заменяет часть обычных фагов на мутантов."""
        normal_phages = [(pos, p) for pos, p in dish.phages.items() if not p.is_mutant]
        num_to_replace = max(1, int(len(normal_phages) * 0.1)) # Заменяем 10%

        phages_to_replace = random.sample(normal_phages, k=min(num_to_replace, len(normal_phages)))

        for (r, c), _ in phages_to_replace:
            dish.remove_agent(r, c)
            mutant_phage = Phage(counters=self.cfg.MUTANT_PHAGE_REPERTOIRE, is_mutant=True)
            dish.add_agent(r, c, mutant_phage)

    def _get_neighbors(self, r, c, diagonal=False):
        """Вспомогательная функция для получения координат соседей."""
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        neighbors = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.cfg.GRID_SIZE and 0 <= nc < self.cfg.GRID_SIZE and self.petri_dish_a.grid[nr, nc] != -1:
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
        self.initialize_petri_dishes(init_a=True, init_b=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

        im1 = axes[0].imshow(self.im_data_a, interpolation='none')
        axes[0].set_title(self.cfg.PRODUCER_1_PARAMS['name'])
        axes[0].set_xticks([]); axes[0].set_yticks([])

        im2 = axes[1].imshow(self.im_data_b, interpolation='none')
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
        ax_plot.set_ylim(1, self.cfg.GRID_SIZE**2)
        ax_plot.axvline(self.cfg.FRAME_ADD_PHAGE, color='gray', linestyle='--', lw=1, label='Внесение фагов')
        if self.cfg.FRAME_MUTANT_PHAGE_APPEARS < self.cfg.N_FRAMES:
             ax_plot.axvline(self.cfg.FRAME_MUTANT_PHAGE_APPEARS, color='purple', linestyle='--', lw=1, label='Мутация фага')
        ax_plot.legend(loc='lower left', fontsize='small')
        ax_plot.grid(True, alpha=0.3)
        plt.tight_layout()

        def update(frame):
            """Основная функция обновления для анимации."""
            self.step()

            self._update_im_data(self.petri_dish_a, self.im_data_a)
            self._update_im_data(self.petri_dish_b, self.im_data_b)
            im1.set_data(self.im_data_a)
            im2.set_data(self.im_data_b)

            self.history['t'].append(frame)
            self._collect_history()

            for key, line in lines.items():
                line.set_data(self.history['t'], self.history[key])

            if frame % 20 == 0: print(f"Кадр {frame}/{self.cfg.N_FRAMES}")
            return list(lines.values()) + [im1, im2]

        anim = FuncAnimation(fig, update, frames=self.cfg.N_FRAMES, interval=50, blit=True)
        plt.show()

    def _update_im_data(self, dish, im_data):
        """Обновляет RGBA-массив для отрисовки."""
        im_data.fill(0)
        im_data[:, :, 3] = 1.0

        colors = {
            'ACTIVE': [0.2, 0.8, 0.2, 1.0], 'REFUGE': [0.1, 0.3, 0.1, 1.0],
            'INFECTED': [1.0, 1.0, 0.0, 1.0], 'PHAGE': [1.0, 0.2, 0.2, 1.0],
            'MUTANT_PHAGE': [0.8, 0.0, 1.0, 1.0] # Фиолетовый
        }

        for (r,c), agent in dish.bacteria.items(): im_data[r, c] = colors[agent.state]
        for (r,c), agent in dish.phages.items():
             im_data[r, c] = colors['MUTANT_PHAGE'] if agent.is_mutant else colors['PHAGE']

        mask = dish.grid == -1
        im_data[mask] = [0.1, 0.1, 0.1, 1.0]


    def _collect_history(self):
        """Собирает статистику по популяциям для графика."""
        self.history['b1'].append(len(self.petri_dish_a.bacteria) if self.petri_dish_a.bacteria else 1)
        self.history['p1'].append(len(self.petri_dish_a.phages) if self.petri_dish_a.phages else 1)
        self.history['b2'].append(len(self.petri_dish_b.bacteria) if self.petri_dish_b.bacteria else 1)
        self.history['p2'].append(len(self.petri_dish_b.phages) if self.petri_dish_b.phages else 1)


if __name__ == "__main__":
    config = Config()
    # Пример запуска анимации с мутантом
    # config.FRAME_MUTANT_PHAGE_APPEARS = 250
    sim = Simulation(config, use_headless=False)
    sim.run_animation()
