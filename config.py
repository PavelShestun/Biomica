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

    # --- Эволюционные параметры ---
    N_DEFENSE_TYPES = 10
    COST_PER_DEFENSE = 0.05
    COST_PER_COUNTER = 0.8
    P_MUTATION = 0.01

    # --- Параметры событий ---
    FRAME_ADD_PHAGE = 80
    INITIAL_PHAGE_COUNT = 150
    PHAGE_ATTACK_REPERTOIRE = {2, 5, 7}

    @property
    def PRODUCER_1_PARAMS(self):
        return {
            'name': "Производитель 1 (Устойчивый)",
            'initial_defenses': lambda: set(random.sample(range(self.N_DEFENSE_TYPES), k=random.randint(2, 3))),
            'p_enter_refuge': 0.1,
            'p_leave_refuge': 0.02
        }

    @property
    def PRODUCER_2_PARAMS(self):
        return {
            'name': "Производитель 2 (Уязвимый)",
            'initial_defenses': lambda: set(),
            'p_enter_refuge': 0.001,
            'p_leave_refuge': 0.02
        }
