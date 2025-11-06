# -*- coding: utf-8 -*-
"""
Конфигурационный файл для симуляции фаговой устойчивости.
"""
import random

class Config:
    # --- Параметры среды ---
    GRID_SIZE = 101
    CENTER = GRID_SIZE // 2
    RADIUS = 50
    N_FRAMES = 500

    # --- Начальные условия ---
    N_BACTERIA = 400
    INITIAL_PHAGE_COUNT = 150

    # --- Параметры питательных веществ ---
    INITIAL_NUTRIENT_LEVEL = 1.0
    NUTRIENT_CONSUMPTION = 0.1
    DIFFUSION_COEFFICIENT = 0.1

    # --- Биологические параметры ---
    LATENCY_PERIOD = 8
    BURST_SIZE_MAX = 12
    P_INFECTION = 0.8
    P_PHAGE_DECAY = 0.03
    R_MAX_GROWTH = 0.8
    COST_PER_REFUGE_TRANSITION = 0.01 # Цена за переход в/из убежища

    # --- Эволюционные параметры ---
    N_DEFENSE_TYPES = 10
    DEFENSE_COSTS = {i: 0.03 if i < 5 else 0.08 for i in range(N_DEFENSE_TYPES)}
    COST_PER_COUNTER = 0.8
    P_MUTATION = 0.01

    # --- Параметры событий ---
    FRAME_ADD_PHAGE = 80
    PHAGE_ATTACK_REPERTOIRE = {2, 5, 7}
    FRAME_MUTANT_PHAGE_APPEARS = 9999
    MUTANT_PHAGE_REPERTOIRE = set(range(N_DEFENSE_TYPES))

    # --- Параметры для экспериментов / Продюсеров ---
    # Эти параметры можно изменять для запуска разных сценариев
    P1_INITIAL_DEFENSE_SIZE = 2
    P1_P_ENTER_REFUGE = 0.1
    P1_P_LEAVE_REFUGE = 0.02

    P2_INITIAL_DEFENSE_SIZE = 0
    P2_P_ENTER_REFUGE = 0.001
    P2_P_LEAVE_REFUGE = 0.02


    @property
    def PRODUCER_1_PARAMS(self):
        """Параметры для Продюсера 1. Используют настраиваемые переменные класса."""
        return {
            'name': "Производитель 1 (Устойчивый)",
            'initial_defenses': lambda: set(random.sample(range(self.N_DEFENSE_TYPES), k=min(self.P1_INITIAL_DEFENSE_SIZE, self.N_DEFENSE_TYPES))),
            'p_enter_refuge': self.P1_P_ENTER_REFUGE,
            'p_leave_refuge': self.P1_P_LEAVE_REFUGE
        }

    @property
    def PRODUCER_2_PARAMS(self):
        """Параметры для Продюсера 2. Используют настраиваемые переменные класса."""
        # Для P2, если defense size = 0, всегда возвращаем пустое множество
        initial_defenses_func = lambda: set() if self.P2_INITIAL_DEFENSE_SIZE == 0 else set(random.sample(range(self.N_DEFENSE_TYPES), k=min(self.P2_INITIAL_DEFENSE_SIZE, self.N_DEFENSE_TYPES)))
        return {
            'name': "Производитель 2 (Уязвимый)",
            'initial_defenses': initial_defenses_func,
            'p_enter_refuge': self.P2_P_ENTER_REFUGE,
            'p_leave_refuge': self.P2_P_LEAVE_REFUGE
        }
