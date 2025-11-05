# -*- coding: utf-8 -*-
"""
Интерактивная вычислительная модель для демонстрации негенетических факторов
устойчивости заквасочных культур к бактериофагам.

Модель реализует две ключевые гипотезы:
1.  "Цена Устойчивости": Бактерии могут иметь "репертуар защит", но каждая
    защита снижает скорость роста. Фаги имеют "репертуар ключей" для
    преодоления защит.
2.  "Клетки в Убежище": Бактерии могут переходить в метаболически неактивное
    ("спящее") состояние, в котором они неуязвимы для фагов.

Модель сравнивает два сценария (двух производителей), отличающихся начальным
состоянием культуры и склонностью к переходу в "убежище".
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import random
from collections import defaultdict

# --- 1. Глобальные параметры симуляции ---

# Параметры среды
GRID_SIZE = 101  # Размер сетки (нечетный для удобства центрирования)
CENTER = GRID_SIZE // 2
RADIUS = 50      # Радиус чашки Петри
N_FRAMES = 500   # Количество кадров (шагов времени) для анимации

# Типы объектов на сетке (для внутреннего использования)
OUT_OF_BOUNDS = -1

# Биологические параметры
LATENCY_PERIOD = 8       # Шагов от заражения до лизиса (гибели) клетки
BURST_SIZE_MAX = 12      # Максимальное количество фагов, высвобождаемых при лизисе
P_INFECTION = 0.8        # Вероятность заражения при контакте фага с уязвимой бактерией
P_PHAGE_DECAY = 0.03     # Вероятность естественного распада (исчезновения) фага за один шаг
R_MAX_GROWTH = 0.8       # Максимальная скорость роста бактерии (без систем защиты)

# Параметры для Гипотезы 1: "Цена Устойчивости"
N_DEFENSE_TYPES = 10     # Общее количество возможных систем защиты (например, 0, 1, ..., 9)
COST_PER_DEFENSE = 0.05  # Насколько снижается скорость роста за каждую активную систему защиты
COST_PER_COUNTER = 0.8   # Насколько снижается burst size за каждый "ключ" в репертуаре фага
P_MUTATION = 0.01        # Вероятность мутации (добавления/удаления системы защиты) при делении

# Параметры симуляционных событий
FRAME_ADD_PHAGE = 80     # На каком шаге времени вносится фаговая инфекция
INITIAL_PHAGE_COUNT = 150# Начальное количество фаговых частиц
PHAGE_ATTACK_REPERTOIRE = {2, 5, 7} # "Ключи", которыми обладают фаги в данной симуляции

# --- 2. Параметры для двух сценариев (Производителей) ---

PRODUCER_1_PARAMS = {
    'name': "Производитель 1 (Устойчивый)",
    # Начальная популяция имеет 2-3 случайные системы защиты
    'initial_defenses': lambda: set(random.sample(range(N_DEFENSE_TYPES), k=random.randint(2, 3))),
    # Гипотеза 2: Высокая склонность переходить в "спящее" состояние
    'p_enter_refuge': 0.1,
    'p_leave_refuge': 0.02
}

PRODUCER_2_PARAMS = {
    'name': "Производитель 2 (Уязвимый)",
    # Начальная популяция не имеет защит, что обеспечивает максимальную скорость роста
    'initial_defenses': lambda: set(),
    # Гипотеза 2: Низкая склонность к переходу в "убежище"
    'p_enter_refuge': 0.001,
    'p_leave_refuge': 0.02
}


# --- 3. Классы Агентов ---

class Bacterium:
    """
    Класс для агента-бактерии. Хранит свое состояние, репертуар защит
    и таймер, если клетка заражена.
    """
    def __init__(self, defenses, state='ACTIVE'):
        self.state = state  # 'ACTIVE', 'REFUGE' (в убежище), 'INFECTED' (заражена)
        self.defenses = defenses # Множество (set) активных систем защиты, например {2, 5}
        self.infection_timer = 0 # Таймер до лизиса
        self.infected_by = None # Хранит объект фага, который заразил клетку

    def is_vulnerable_to(self, phage):
        """
        Проверяет, может ли данный фаг заразить эту бактерию.
        Возвращает True, если репертуар "ключей" фага является надмножеством
        репертуара защит бактерии.
        """
        return phage.counters.issuperset(self.defenses)

class Phage:
    """Класс для агента-фага. Хранит свой репертуар "ключей"."""
    def __init__(self, counters):
        self.counters = counters # Множество (set) "ключей", например {2, 5, 7}

# --- 4. Функции симуляции ---

def get_neighbors(r, c, diagonal=False):
    """Вспомогательная функция для получения координат соседних клеток."""
    deltas = [(-1,0), (1,0), (0,-1), (0,1)]
    if diagonal:
        deltas.extend([(-1,-1), (-1,1), (1,-1), (1,1)])
    neighbors = []
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            neighbors.append((nr, nc))
    return neighbors

def mutate_repertoire(repertoire):
    """
    Реализует мутацию репертуара защит бактерии при делении.
    С вероятностью P_MUTATION добавляет или удаляет одну случайную систему защиты.
    """
    new_repertoire = repertoire.copy()
    if random.random() < P_MUTATION:
        # 50/50 шанс добавить или удалить ген
        if random.random() < 0.5 and len(new_repertoire) > 0:
            new_repertoire.remove(random.choice(list(new_repertoire)))
        elif len(new_repertoire) < N_DEFENSE_TYPES:
            possible_additions = list(set(range(N_DEFENSE_TYPES)) - new_repertoire)
            if possible_additions:
                new_repertoire.add(random.choice(possible_additions))
    return new_repertoire

def simulation_step(grid_of_agents, K, params):
    """
    Выполняет один шаг симуляции, ОБЪЕДИНЯЯ обе гипотезы.
    `grid_of_agents`: 2D numpy массив, хранящий объекты агентов.
    `K`: Максимальная емкость среды (количество бактерий).
    `params`: Словарь с параметрами для конкретного производителя.
    """
    new_grid = grid_of_agents.copy()

    # Координаты обрабатываются в случайном порядке, чтобы избежать артефактов
    all_coords = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    random.shuffle(all_coords)

    # Подсчет общей популяции для моделирования логистического роста
    total_bacteria = np.sum([isinstance(agent, Bacterium) for agent in grid_of_agents.ravel()])

    for r, c in all_coords:
        agent = grid_of_agents[r, c]
        if agent is None or agent == OUT_OF_BOUNDS:
            continue

        # --- Логика для БАКТЕРИЙ ---
        if isinstance(agent, Bacterium):
            # --- Гипотеза 2: "Клетки в Убежище" ---
            if agent.state == 'REFUGE': # Пробуждение
                if random.random() < params['p_leave_refuge']: agent.state = 'ACTIVE'
            elif agent.state == 'ACTIVE': # Переход в убежище
                if random.random() < params['p_enter_refuge']: agent.state = 'REFUGE'

            # --- Гипотеза 1: "Цена Устойчивости" ---
            if agent.state == 'ACTIVE': # Рост и деление (только для активных)
                # Скорость роста зависит от количества защит и общей популяции
                growth_prob = (R_MAX_GROWTH - COST_PER_DEFENSE * len(agent.defenses))
                growth_prob *= (1 - total_bacteria / K)

                if random.random() < growth_prob:
                    empty_neighbors = [pos for pos in get_neighbors(r,c) if new_grid[pos] is None]
                    if empty_neighbors:
                        nr, nc = random.choice(empty_neighbors)
                        new_defenses = mutate_repertoire(agent.defenses)
                        new_grid[nr, nc] = Bacterium(defenses=new_defenses, state='ACTIVE')

            elif agent.state == 'INFECTED': # Лизис
                agent.infection_timer -= 1
                if agent.infection_timer <= 0:
                    new_grid[r, c] = None # Клетка погибает
                    phage_parent = agent.infected_by
                    # "Цена" репертуара фага влияет на размер потомства
                    burst_size = int(BURST_SIZE_MAX - COST_PER_COUNTER * len(phage_parent.counters))

                    empty_neighbors = [pos for pos in get_neighbors(r,c, diagonal=True) if new_grid[pos] is None]
                    random.shuffle(empty_neighbors)
                    for _ in range(burst_size):
                        if not empty_neighbors: break
                        nr, nc = empty_neighbors.pop()
                        new_grid[nr, nc] = Phage(counters=phage_parent.counters)

        # --- Логика для ФАГОВ ---
        elif isinstance(agent, Phage):
            if random.random() < P_PHAGE_DECAY: # Распад
                new_grid[r, c] = None
                continue

            # Поиск и заражение уязвимой бактерии по соседству
            potential_targets = []
            for nr, nc in get_neighbors(r, c):
                neighbor = grid_of_agents[nr, nc]
                if isinstance(neighbor, Bacterium) and neighbor.state == 'ACTIVE' and neighbor.is_vulnerable_to(agent):
                    potential_targets.append((nr, nc))

            if potential_targets:
                target_r, target_c = random.choice(potential_targets)
                target_bacterium = new_grid[target_r, target_c]

                if random.random() < P_INFECTION:
                    target_bacterium.state = 'INFECTED'
                    target_bacterium.infection_timer = LATENCY_PERIOD
                    target_bacterium.infected_by = agent
                    new_grid[r, c] = None # Фаг адсорбировался
            else: # Движение в случайную пустую клетку
                empty_neighbors = [pos for pos in get_neighbors(r,c) if new_grid[pos] is None]
                if empty_neighbors:
                    nr, nc = random.choice(empty_neighbors)
                    new_grid[nr, nc], new_grid[r, c] = new_grid[r, c], None

    return new_grid

def setup_simulation(params):
    """
    Создает начальное состояние сетки для симуляции.
    """
    grid = np.full((GRID_SIZE, GRID_SIZE), None, dtype=object)

    y, x = np.ogrid[-CENTER:GRID_SIZE - CENTER, -CENTER:GRID_SIZE - CENTER]
    mask = x * x + y * y > RADIUS * RADIUS
    grid[mask] = OUT_OF_BOUNDS

    # Создаем небольшую начальную колонию
    colony_radius = 5
    for r_offset in range(-colony_radius, colony_radius + 1):
        for c_offset in range(-colony_radius, colony_radius + 1):
            if r_offset**2 + c_offset**2 <= colony_radius**2:
                r, c = CENTER + r_offset, CENTER + c_offset
                defenses = params['initial_defenses']()
                grid[r, c] = Bacterium(defenses=defenses, state='ACTIVE')
    return grid

def add_phages(grid):
    """Добавляет начальное количество фагов в пустые ячейки сетки."""
    empty_coords = np.argwhere(grid == None)
    if len(empty_coords) > INITIAL_PHAGE_COUNT:
        indices = np.random.choice(len(empty_coords), INITIAL_PHAGE_COUNT, replace=False)
        for idx in indices:
            r, c = empty_coords[idx]
            grid[r, c] = Phage(counters=PHAGE_ATTACK_REPERTOIRE)

# --- 5. Визуализация ---

# Создаем RGBA-массивы для прямого контроля над цветом и производительности
im_data1 = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
im_data2 = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)

def update_im_data(grid, im_data):
    """
    Напрямую заполняет RGBA-массив цветов для одной чашки Петри,
    что намного быстрее, чем перерисовывать imshow с нуля.
    """
    colors = {
        'OUT_OF_BOUNDS': [0.1, 0.1, 0.1, 1.0],
        'EMPTY':         [0.0, 0.0, 0.0, 1.0],
        'ACTIVE':        [0.2, 0.8, 0.2, 1.0],  # Ярко-зеленый
        'REFUGE':        [0.1, 0.3, 0.1, 1.0],  # Темно-зеленый (тусклый)
        'INFECTED':      [1.0, 1.0, 0.0, 1.0],  # Желтый
        'PHAGE':         [1.0, 0.2, 0.2, 1.0],  # Красный
    }

    im_data[:,:] = colors['EMPTY'] # Фон по умолчанию - черный

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            agent = grid[r, c]
            if agent == OUT_OF_BOUNDS: im_data[r, c] = colors['OUT_OF_BOUNDS']
            elif isinstance(agent, Bacterium): im_data[r, c] = colors[agent.state]
            elif isinstance(agent, Phage): im_data[r, c] = colors['PHAGE']

def update(frame):
    """
    Основная функция обновления, вызываемая для каждого кадра анимации.
    """
    global grid1, grid2

    # Внесение фагов в систему в заданный момент
    if frame == FRAME_ADD_PHAGE:
        add_phages(grid1)
        add_phages(grid2)

    # Выполнение одного шага симуляции для каждой модели
    grid1 = simulation_step(grid1, K, PRODUCER_1_PARAMS)
    grid2 = simulation_step(grid2, K, PRODUCER_2_PARAMS)

    # Обновление изображений на основе новых состояний сеток
    update_im_data(grid1, im_data1)
    update_im_data(grid2, im_data2)
    im1.set_data(im_data1)
    im2.set_data(im_data2)

    # Сбор статистики и обновление графика
    history['t'].append(frame)
    b1_count, p1_count = 0, 0
    for agent in grid1.ravel():
        if isinstance(agent, Bacterium): b1_count += 1
        elif isinstance(agent, Phage): p1_count += 1

    b2_count, p2_count = 0, 0
    for agent in grid2.ravel():
        if isinstance(agent, Bacterium): b2_count += 1
        elif isinstance(agent, Phage): p2_count += 1

    history['b1'].append(b1_count if b1_count > 0 else 1) # Используем 1 для log-шкалы
    history['p1'].append(p1_count if p1_count > 0 else 1)
    history['b2'].append(b2_count if b2_count > 0 else 1)
    history['p2'].append(p2_count if p2_count > 0 else 1)

    line_b1.set_data(history['t'], history['b1'])
    line_p1.set_data(history['t'], history['p1'])
    line_b2.set_data(history['t'], history['b2'])
    line_p2.set_data(history['t'], history['p2'])

    if frame % 20 == 0:
        print(f"Кадр {frame}/{N_FRAMES}")

    return [im1, im2, line_b1, line_p1, line_b2, line_p2]


# --- Основной блок выполнения ---
if __name__ == "__main__":
    # 1. Настройка фигуры для визуализации
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # 2. Инициализация симуляций для обоих производителей
    grid1 = setup_simulation(PRODUCER_1_PARAMS)
    grid2 = setup_simulation(PRODUCER_2_PARAMS)
    # Емкость среды (K) одинакова для обеих чашек
    K = np.sum(grid1 != OUT_OF_BOUNDS)

    # 3. Настройка визуализации для Чашки 1
    im1 = axes[0].imshow(im_data1, interpolation='none')
    axes[0].set_title(PRODUCER_1_PARAMS['name'])
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # 4. Настройка визуализации для Чашки 2
    im2 = axes[1].imshow(im_data2, interpolation='none')
    axes[1].set_title(PRODUCER_2_PARAMS['name'])
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # 5. Настройка графика динамики популяций
    history = defaultdict(list)
    ax_plot = axes[2]
    line_b1, = ax_plot.plot([], [], color='#4285F4', label='Бактерии 1')
    line_p1, = ax_plot.plot([], [], color='#A5C8FF', linestyle=':', label='Фаги 1')
    line_b2, = ax_plot.plot([], [], color='#EA4335', label='Бактерии 2')
    line_p2, = ax_plot.plot([], [], color='#FADADD', linestyle=':', label='Фаги 2')

    ax_plot.set_title("Динамика популяций")
    ax_plot.set_xlabel("Время (шаги симуляции)")
    ax_plot.set_ylabel("Численность (log шкала)")
    ax_plot.set_xlim(0, N_FRAMES)
    ax_plot.set_yscale('log')
    ax_plot.set_ylim(1, K * 1.5)
    ax_plot.axvline(FRAME_ADD_PHAGE, color='yellow', linestyle='--', lw=1, label='Внесение фагов')
    ax_plot.legend(loc='lower left', fontsize='small')
    ax_plot.grid(True, alpha=0.3)

    plt.tight_layout()

    # 6. Создание и запуск анимации
    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=80, blit=True)

    # Для сохранения анимации в видеофайл (требуется ffmpeg)
    # print("Сохранение анимации... Это может занять несколько минут.")
    # anim.save('phage_dynamics_simulation.mp4', writer='ffmpeg', fps=15, dpi=150)
    # print("Анимация сохранена.")

    plt.show()
