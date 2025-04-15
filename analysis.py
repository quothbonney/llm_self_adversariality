import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any # For type hinting

# --- Helper Function: Data Processing ---

def process_belief_data(belief_data: Optional[Dict[int, Dict[str, Any]]], run_name: str = "Unknown") -> pd.DataFrame:
    """
    Converts a dictionary of belief data per step into a Pandas DataFrame.

    Args:
        belief_data: A dictionary where keys are step numbers and values are
                     dictionaries containing 'Generated', 'True_Logprob', 'False_Logprob'.
                     Can be None.
        run_name: A string identifier for the data source (e.g., "Clean", "Poisoned").

    Returns:
        A Pandas DataFrame with columns: 'step', 'generated', 'belief_score'
        (normalized [0-1] where 1=True), 'prob_generated', 'belief_numeric'
        (1 for True, 0 for False, NaN otherwise). Returns an empty DataFrame
        if input is None or empty.
    """
    if not belief_data:
        print(f"INFO: No belief data provided for {run_name} run.")
        return pd.DataFrame(columns=[
            'step', 'generated', 'belief_score', 'prob_generated', 'belief_numeric'
        ])

    steps = sorted(list(belief_data.keys()))
    processed_data = []

    for step in steps:
        data = belief_data.get(step)
        if not data:
            print(f"WARNING: Missing data for step {step} in {run_name} run.")
            continue

        generated = data.get('Generated')
        true_lp = data.get('True_Logprob')
        false_lp = data.get('False_Logprob')

        # Convert logprobs to probabilities (handle None and ensure non-positive logprobs)
        true_prob = np.exp(min(true_lp, 0.0)) if true_lp is not None else np.nan
        false_prob = np.exp(min(false_lp, 0.0)) if false_lp is not None else np.nan

        # Calculate normalized belief score [0-1] where 1 means "True"
        belief_score = np.nan
        if not pd.isna(true_prob) and not pd.isna(false_prob):
            # Avoid division by zero if both probabilities are effectively zero
            denominator = true_prob + false_prob
            if denominator > 1e-9: # Use a small epsilon
                 belief_score = true_prob / denominator
            elif true_prob > false_prob: # If both are tiny, lean towards the larger one
                 belief_score = 1.0
            else:
                 belief_score = 0.0
        elif not pd.isna(true_prob):
            belief_score = true_prob # Only True prob available
        elif not pd.isna(false_prob):
            belief_score = 1.0 - false_prob # Only False prob available

        # Calculate the probability of the *actually generated* token
        prob_gen = np.nan
        if generated == 'True':
            prob_gen = true_prob
        elif generated == 'False':
            # We want P(False), which is approx. (1 - belief_score) if normalized
            # Or directly use false_prob if available
             prob_gen = false_prob # Use the direct probability if available

        # Assign numeric value for easier plotting/filtering
        belief_numeric = np.nan
        if generated == 'True':
            belief_numeric = 1
        elif generated == 'False':
            belief_numeric = 0

        processed_data.append({
            'step': step,
            'generated': generated,
            'belief_score': belief_score,
            'prob_generated': prob_gen,
            'belief_numeric': belief_numeric
        })

    if not processed_data:
         print(f"INFO: No valid steps processed for {run_name} run.")
         return pd.DataFrame(columns=[
            'step', 'generated', 'belief_score', 'prob_generated', 'belief_numeric'
         ])

    return pd.DataFrame(processed_data)


# --- Main Function: Visualization ---

def plot_belief_analysis(
    clean_belief_data: Optional[Dict[int, Dict[str, Any]]],
    poisoned_belief_data: Optional[Dict[int, Dict[str, Any]]],
    belief_query: str,
    poison_step_indices: Optional[List[int]] = None,
    use_seaborn: bool = True
) -> plt.Figure:
    """
    Generates a two-panel plot comparing belief trends for clean and poisoned runs.

    Args:
        clean_belief_data: Raw belief data dictionary for the clean run.
        poisoned_belief_data: Raw belief data dictionary for the poisoned run.
        belief_query: The query string used for the belief task (for plot title).
        poison_step_indices: A list of step indices where poisoning was applied.
        use_seaborn: If True, attempts to use Seaborn styling. Falls back otherwise.

    Returns:
        matplotlib.figure.Figure: The generated plot figure object. The caller is
                                  responsible for showing or saving the figure.
    """
    # --- Plot Style Setup ---
    if use_seaborn:
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            print("Using Seaborn style for plots.")
        except ImportError:
            plt.style.use('seaborn-v0_8-whitegrid') # Fallback style
            print("Seaborn not found, using default Matplotlib 'seaborn-v0_8-whitegrid' style.")
    else:
         # Use a default non-seaborn style if preferred
         plt.style.use('default') # Or another specific style
         print("Using default Matplotlib style.")


    print("\n--- Starting Belief Analysis Visualization ---")

    # --- Process Data ---
    print("Processing Clean Data...")
    df_clean = process_belief_data(clean_belief_data, "Clean")
    print(f"Processed Clean Data. Shape: {df_clean.shape}")

    print("\nProcessing Poisoned Data...")
    df_poisoned = process_belief_data(poisoned_belief_data, "Poisoned")
    print(f"Processed Poisoned Data. Shape: {df_poisoned.shape}")

    plot_clean = not df_clean.empty
    plot_poisoned = not df_poisoned.empty

    if not plot_clean and not plot_poisoned:
        print("\nERROR: No valid data available to generate plots. Returning empty figure.")
        # Return an empty figure or raise an error
        fig, ax = plt.subplots(2, 1, figsize=(14, 10))
        ax[0].set_title("No Data Available")
        ax[1].set_title("No Data Available")
        fig.tight_layout()
        return fig # Return the empty figure

    # --- Visualization ---
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    marker_size = 120 # Adjusted default marker size

    # --- Plot 1: Belief Score (Log Probability of True) ---
    ax[0].set_title(f'Belief Score (P(True)) for Query: "{belief_query}"')
    ax[0].set_ylabel("Belief Score (P(True))")
    ax[0].set_ylim(-0.05, 1.05) # Belief score is naturally [0, 1]

    # Filter data for True/False generated tokens
    df_clean_true = df_clean[df_clean['belief_numeric'] == 1] if plot_clean else pd.DataFrame()
    df_clean_false = df_clean[df_clean['belief_numeric'] == 0] if plot_clean else pd.DataFrame()
    df_poisoned_true = df_poisoned[df_poisoned['belief_numeric'] == 1] if plot_poisoned else pd.DataFrame()
    df_poisoned_false = df_poisoned[df_poisoned['belief_numeric'] == 0] if plot_poisoned else pd.DataFrame()

    # Plot Clean Run (Belief Score)
    if plot_clean:
        ax[0].plot(df_clean['step'], df_clean['belief_score'], marker='o', markersize=5, linestyle='-',
                   label='Clean Run (Belief)', color='skyblue', zorder=2)
        if not df_clean_true.empty:
            ax[0].scatter(df_clean_true['step'], df_clean_true['belief_score'],
                         marker='P', color='blue', s=marker_size, label='Clean: Gen="True"', zorder=3, edgecolors='black', linewidth=0.5)
        if not df_clean_false.empty:
            ax[0].scatter(df_clean_false['step'], df_clean_false['belief_score'],
                         marker='X', color='blue', s=marker_size, label='Clean: Gen="False"', zorder=3, edgecolors='black', linewidth=0.5)

    # Plot Poisoned Run (Belief Score)
    if plot_poisoned:
        ax[0].plot(df_poisoned['step'], df_poisoned['belief_score'], marker='o', markersize=5, linestyle='--',
                   label='Poisoned Run (Belief)', color='salmon', zorder=2)
        if not df_poisoned_true.empty:
            ax[0].scatter(df_poisoned_true['step'], df_poisoned_true['belief_score'],
                         marker='P', color='red', s=marker_size, label='Poisoned: Gen="True"', zorder=3, edgecolors='black', linewidth=0.5)
        if not df_poisoned_false.empty:
            ax[0].scatter(df_poisoned_false['step'], df_poisoned_false['belief_score'],
                         marker='X', color='red', s=marker_size, label='Poisoned: Gen="False"', zorder=3, edgecolors='black', linewidth=0.5)

    # Add vertical lines for poison steps
    if poison_step_indices:
        sorted_indices = sorted(list(set(poison_step_indices))) # Ensure unique and sorted
        for i, ps in enumerate(sorted_indices):
            line_label = f'Poison Step ({ps})' if i == 0 else "_nolegend_"
            ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=2, label=line_label, zorder=1)

    ax[0].legend(loc='best', fontsize='small')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2: Generated Token Probability ---
    ax[1].set_title('Probability of Generated Answer (P(Generated Token))')
    ax[1].set_xlabel("Reasoning Step")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(-0.05, 1.05) # Probability is naturally [0, 1]

    # Plot Clean Run (Generated Probability)
    if plot_clean:
        # Use the pre-calculated prob_generated
        ax[1].plot(df_clean['step'], df_clean['prob_generated'], marker='o', markersize=5, linestyle='-',
                   label='Clean Run (P(Gen))', color='skyblue', zorder=2)
        if not df_clean_true.empty:
            ax[1].scatter(df_clean_true['step'], df_clean_true['prob_generated'],
                         marker='P', color='blue', s=marker_size, label='Clean: Gen="True"', zorder=3, edgecolors='black', linewidth=0.5)
        if not df_clean_false.empty:
            ax[1].scatter(df_clean_false['step'], df_clean_false['prob_generated'],
                         marker='X', color='blue', s=marker_size, label='Clean: Gen="False"', zorder=3, edgecolors='black', linewidth=0.5)

    # Plot Poisoned Run (Generated Probability)
    if plot_poisoned:
        ax[1].plot(df_poisoned['step'], df_poisoned['prob_generated'], marker='o', markersize=5, linestyle='--',
                   label='Poisoned Run (P(Gen))', color='salmon', zorder=2)
        if not df_poisoned_true.empty:
            ax[1].scatter(df_poisoned_true['step'], df_poisoned_true['prob_generated'],
                         marker='P', color='red', s=marker_size, label='Poisoned: Gen="True"', zorder=3, edgecolors='black', linewidth=0.5)
        if not df_poisoned_false.empty:
            ax[1].scatter(df_poisoned_false['step'], df_poisoned_false['prob_generated'],
                         marker='X', color='red', s=marker_size, label='Poisoned: Gen="False"', zorder=3, edgecolors='black', linewidth=0.5)


    # Add vertical lines for poison steps
    if poison_step_indices:
         sorted_indices = sorted(list(set(poison_step_indices)))
         for i, ps in enumerate(sorted_indices):
            line_label = f'Poison Step ({ps})' if i == 0 else "_nolegend_"
            ax[1].axvline(x=ps, color='grey', linestyle=':', linewidth=2, label=line_label, zorder=1)

    ax[1].legend(loc='best', fontsize='small')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- General Plot Adjustments ---
    max_step = 0
    if plot_poisoned:
        max_step = max(max_step, df_poisoned['step'].max())
    if plot_clean:
        max_step = max(max_step, df_clean['step'].max())

    if max_step > 0:
        # Adjust x-ticks for readability
        tick_step = max(1, int(np.ceil(max_step / 10))) # Aim for ~10 ticks
        plt.xticks(np.arange(0, max_step + 1, tick_step))
    else: # Handle case with only step 0 or no steps
         plt.xticks([0])

    plt.xlim(left=-0.5) # Add a little padding on the left
    if max_step > 0 :
        plt.xlim(right=max_step + 0.5) # Add padding on the right

    fig.tight_layout() # Adjust layout to prevent overlap

    print("--- Visualization Complete ---")

    return fig # Return the figure object