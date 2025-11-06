import pandas as pd
import numpy as np
from main import Simulation
from config import Config
import time
from tqdm import tqdm
import itertools

def run_experiment1():
    """
    Runs Experiment 1: Find Optimal Defense Repertoire.
    Iterates through different initial defense repertoire sizes to find the one
    that maximizes final bacterial population.
    """
    print("Starting Experiment 1: Find Optimal Defense Repertoire...")
    cfg = Config()
    # Use smaller, faster parameters for this experiment
    cfg.GRID_SIZE = 61
    cfg.N_BACTERIA = 150
    cfg.INITIAL_PHAGE_COUNT = 100
    cfg.N_FRAMES = 150
    cfg.FRAME_ADD_PHAGE = 10
    cfg.FRAME_MUTANT_PHAGE_APPEARS = 9999 # Mutant disabled for this experiment

    # --- Experimental Parameters ---
    repertoire_sizes = range(0, 7)
    n_runs_per_setting = 5

    # --- Data Collection ---
    results = []
    run_list = list(itertools.product(repertoire_sizes, range(n_runs_per_setting)))

    for rep_size, run_id in tqdm(run_list, desc="Testing Repertoire Sizes"):
        cfg.P1_INITIAL_DEFENSE_SIZE = rep_size

        sim = Simulation(cfg, use_headless=True)
        sim.initialize_petri_dishes(init_a=True, init_b=True) # Use both for comparison in raw data

        for _ in range(cfg.N_FRAMES):
            sim.step()

        final_b1 = len(sim.petri_dish_a.bacteria)
        final_p1 = len(sim.petri_dish_a.phages)
        final_b2 = len(sim.petri_dish_b.bacteria)
        final_p2 = len(sim.petri_dish_b.phages)

        results.append({
            'combo_id': rep_size,
            'run_id': run_id,
            'P1_INITIAL_DEFENSE_SIZE': rep_size,
            'p1_enter_refuge': cfg.P1_P_ENTER_REFUGE, # record this for clarity
            'final_b1': final_b1,
            'final_p1': final_p1,
            'final_b2': final_b2,
            'final_p2': final_p2
        })

    df = pd.DataFrame(results)
    df.to_csv("experiment1_results.csv", index=False)
    print("Experiment 1 complete. Results saved to experiment1_results.csv")

def run_experiment2():
    """
    Runs Experiment 2: Compare Survival Strategies.
    Compares three strategies: Refuge-Only, Repertoire-Only, and Hybrid.
    """
    print("Starting Experiment 2: Compare Survival Strategies...")
    cfg = Config()
    cfg.GRID_SIZE = 71
    cfg.N_BACTERIA = 200
    cfg.INITIAL_PHAGE_COUNT = 150
    cfg.N_FRAMES = 200
    cfg.FRAME_ADD_PHAGE = 10
    cfg.FRAME_MUTANT_PHAGE_APPEARS = 100
    cfg.MUTANT_PHAGE_REPERTOIRE = {3, 4}

    scenarios = {
        "Refuge-Only": {"initial_defense_size": 0, "p_enter_refuge": 0.1},
        "Repertoire-Only": {"initial_defense_size": 2, "p_enter_refuge": 0.0},
        "Hybrid": {"initial_defense_size": 2, "p_enter_refuge": 0.1}
    }

    all_results = []
    for name, params in scenarios.items():
        print(f"Running scenario: {name}...")
        cfg.P1_INITIAL_DEFENSE_SIZE = params["initial_defense_size"]
        cfg.P1_P_ENTER_REFUGE = params["p_enter_refuge"]

        sim = Simulation(cfg, use_headless=True)
        sim.initialize_petri_dishes(init_a=True, init_b=False)

        for frame in tqdm(range(cfg.N_FRAMES), desc=f"Simulating {name}"):
            sim.step()
            n_bacteria = len(sim.petri_dish_a.bacteria)
            n_phages = len(sim.petri_dish_a.phages)
            all_results.append({"frame": frame, "scenario": name, "n_bacteria": n_bacteria, "n_phages": n_phages})
            if n_bacteria == 0:
                for i in range(frame + 1, cfg.N_FRAMES):
                    all_results.append({"frame": i, "scenario": name, "n_bacteria": 0, "n_phages": n_phages})
                break

    df = pd.DataFrame(all_results)
    df.to_csv("experiment2_results.csv", index=False)
    print("Experiment 2 complete. Results saved to experiment2_results.csv")

def run_experiment3():
    """
    Runs Experiment 3: Mutant Phage Stress-Test.
    Tests the Hybrid strategy against a mutant phage that appears at different times.
    """
    print("Starting Experiment 3: Mutant Phage Stress-Test...")
    cfg = Config()
    cfg.GRID_SIZE = 61
    cfg.N_BACTERIA = 150
    cfg.INITIAL_PHAGE_COUNT = 100
    cfg.N_FRAMES = 250
    cfg.FRAME_ADD_PHAGE = 10
    cfg.P1_INITIAL_DEFENSE_SIZE = 2
    cfg.P1_P_ENTER_REFUGE = 0.1
    cfg.MUTANT_PHAGE_REPERTOIRE = {3, 4}

    mutant_appearance_frames = np.linspace(15, 240, 10, dtype=int)
    n_runs_per_setting = 3
    results = []
    run_list = list(itertools.product(mutant_appearance_frames, range(n_runs_per_setting)))

    for frame_mutant_appears, run_id in tqdm(run_list, desc="Stress-Testing Hybrid Strategy"):
        cfg.FRAME_MUTANT_PHAGE_APPEARS = frame_mutant_appears
        sim = Simulation(cfg, use_headless=True)
        sim.initialize_petri_dishes(init_a=True, init_b=False)
        for _ in range(cfg.N_FRAMES):
            sim.step()
        final_bacteria_count = len(sim.petri_dish_a.bacteria)
        results.append({'mutant_appearance_frame': frame_mutant_appears, 'run_id': run_id, 'final_bacteria_count': final_bacteria_count})

    df = pd.DataFrame(results)
    df.to_csv("experiment3_results.csv", index=False)
    print("Experiment 3 complete. Results saved to experiment3_results.csv")


if __name__ == "__main__":
    start_time = time.time()

    # --- Instructions for running experiments ---
    # Uncomment the function for the experiment you want to run.
    # It is recommended to run them one at a time.

    # print("--- Running Experiment 1 ---")
    # run_experiment1()

    # print("\n--- Running Experiment 2 ---")
    # run_experiment2()

    print("\n--- Running Experiment 3 ---")
    run_experiment3()

    end_time = time.time()
    print(f"\nTotal script duration: {end_time - start_time:.2f} seconds")
