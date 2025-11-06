import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_experiment1(filename="experiment1_results.csv"):
    """
    Generates and saves a plot for the results of Experiment 1.
    Aggregates raw run data and plots final bacteria count vs. initial repertoire size.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    df = pd.read_csv(filename)
    agg_df = df.groupby('P1_INITIAL_DEFENSE_SIZE')['final_b1'].agg(['mean', 'std']).reset_index()
    agg_df.rename(columns={'mean': 'final_bacteria_count_mean', 'std': 'final_bacteria_count_std'}, inplace=True)
    agg_df['final_bacteria_count_std'].fillna(0, inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=agg_df, x='P1_INITIAL_DEFENSE_SIZE', y='final_bacteria_count_mean', ax=ax, marker='o', label='Mean Final Bacteria Count')
    ax.fill_between(agg_df['P1_INITIAL_DEFENSE_SIZE'], agg_df['final_bacteria_count_mean'] - agg_df['final_bacteria_count_std'], agg_df['final_bacteria_count_mean'] + agg_df['final_bacteria_count_std'], alpha=0.2, label='Standard Deviation')
    ax.set_title('Experiment 1: Optimal Defense Repertoire Size')
    ax.set_xlabel('Initial Number of Active Defense Systems')
    ax.set_ylabel('Final Bacteria Population Count')
    ax.legend()
    ax.grid(True)
    if not agg_df.empty:
        peak_idx = agg_df['final_bacteria_count_mean'].idxmax()
        peak_size = agg_df.loc[peak_idx, 'P1_INITIAL_DEFENSE_SIZE']
        peak_value = agg_df.loc[peak_idx, 'final_bacteria_count_mean']
        ax.annotate(f'Optimal Size: {int(peak_size)}', xy=(peak_size, peak_value), xytext=(peak_size, peak_value + max(5, peak_value*0.05)), arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    plt.savefig("experiment1_plot.png")
    print("Experiment 1 plot saved to experiment1_plot.png")
    plt.close()

def plot_experiment2(filename="experiment2_results.csv"):
    """
    Generates and saves a plot for the results of Experiment 2.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    df = pd.read_csv(filename)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df, x='frame', y='n_bacteria', hue='scenario', ax=ax, linewidth=2.5)
    ax.set_title('Experiment 2: Comparison of Survival Strategies')
    ax.set_xlabel('Time (Simulation Frames)')
    ax.set_ylabel('Number of Bacteria')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.axvline(10, color='gray', linestyle='--', lw=1.5, label='Phage Introduction')
    ax.axvline(100, color='purple', linestyle='--', lw=1.5, label='Mutant Phage Appears')
    ax.legend(title='Strategy')
    ax.grid(True, which="both", ls="--")
    plt.savefig("experiment2_plot.png")
    print("Experiment 2 plot saved to experiment2_plot.png")
    plt.close()

def plot_experiment3(filename="experiment3_results.csv"):
    """
    Generates and saves a plot for the results of Experiment 3.
    Plots final bacteria count vs. the time of mutant phage appearance.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    df = pd.read_csv(filename)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df,
        x='mutant_appearance_frame',
        y='final_bacteria_count',
        marker='o',
        ci='sd', # Show standard deviation as error bars
        ax=ax
    )

    # Find the minimum survival point
    agg_df = df.groupby('mutant_appearance_frame')['final_bacteria_count'].mean().reset_index()
    min_survival_frame = agg_df.loc[agg_df['final_bacteria_count'].idxmin()]

    ax.set_title('Experiment 3: Mutant Phage Stress-Test on Hybrid Strategy')
    ax.set_xlabel('Time of Mutant Phage Appearance (Simulation Frame)')
    ax.set_ylabel('Final Bacteria Population Count')
    ax.grid(True, which="both", ls="--")

    # Annotate the most vulnerable point
    ax.annotate(
        f'Most Vulnerable Point\n(Frame {int(min_survival_frame.mutant_appearance_frame)})',
        xy=(min_survival_frame.mutant_appearance_frame, min_survival_frame.final_bacteria_count),
        xytext=(min_survival_frame.mutant_appearance_frame + 20, min_survival_frame.final_bacteria_count + 100),
        arrowprops=dict(facecolor='black', shrink=0.05),
        ha='center'
    )

    output_filename = "experiment3_plot.png"
    plt.savefig(output_filename)
    print(f"Experiment 3 plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    print("Generating plots for all completed experiments...")
    plot_experiment1()
    plot_experiment2()
    plot_experiment3()
    print("Done.")
