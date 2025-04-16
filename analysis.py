import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any # For type hinting

# --- Helper Function: Data Processing (Keep as is) ---
# (process_belief_data function remains exactly the same as you provided)
def process_belief_data(belief_data: Optional[Dict[int, Dict[str, Any]]], run_name: str = "Unknown") -> pd.DataFrame:
    """
    Converts a dictionary of belief data per step into a Pandas DataFrame.
    (Function code is omitted here for brevity, assumed to be the same as in your previous prompt)
    """
    if not belief_data:
        # print(f"DEBUG: No belief data provided for {run_name} run.") # Quieter
        return pd.DataFrame(columns=[
            'step', 'generated', 'belief_score', 'prob_generated', 'belief_numeric'
        ])

    steps = sorted(list(belief_data.keys()))
    processed_data = []

    for step in steps:
        data = belief_data.get(step)
        if not data:
            # print(f"DEBUG: Missing data for step {step} in {run_name} run.") # Quieter
            continue

        generated = data.get('Generated')
        true_lp = data.get('True_Logprob')
        false_lp = data.get('False_Logprob')

        # Convert logprobs to probabilities (handle None and ensure non-positive logprobs)
        true_prob = np.exp(min(true_lp, 0.0)) if true_lp is not None and true_lp <= 0 else np.nan
        false_prob = np.exp(min(false_lp, 0.0)) if false_lp is not None and false_lp <= 0 else np.nan


        # Calculate normalized belief score [0-1] where 1 means "True"
        belief_score = np.nan
        if not pd.isna(true_prob) and not pd.isna(false_prob):
            denominator = true_prob + false_prob
            if denominator > 1e-9: # Use a small epsilon
                 belief_score = true_prob / denominator
            elif true_prob > false_prob: # If both are tiny, lean towards the larger one
                 belief_score = 1.0
            else:
                 belief_score = 0.0
        # Handle cases where only one probability is available (less ideal, but provides some info)
        elif not pd.isna(true_prob) and pd.isna(false_prob):
            belief_score = true_prob # Approximation: Assume P(True) dominates if P(False) unknown
        elif pd.isna(true_prob) and not pd.isna(false_prob):
             belief_score = 1.0 - false_prob # Approximation: Assume P(False) dominates if P(True) unknown


        # Calculate the probability of the *actually generated* token
        prob_gen = np.nan
        if generated == 'True':
            prob_gen = true_prob
        elif generated == 'False':
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
        # print(f"DEBUG: No valid steps processed for {run_name} run.") # Quieter
         return pd.DataFrame(columns=[
            'step', 'generated', 'belief_score', 'prob_generated', 'belief_numeric'
         ])

    return pd.DataFrame(processed_data)


# --- Main Function: Visualization (Rewritten for Multiple Runs) ---

def plot_belief_analysis(
    all_run_results: List[Dict[str, Any]], # MODIFIED: Takes list of run results
    belief_query: str,
    poison_step_indices: Optional[List[int]] = None,
    use_seaborn: bool = True
) -> Optional[plt.Figure]: # MODIFIED: Returns Optional Figure
    """
    Generates a two-panel plot showing belief trends for multiple runs.
    Clean runs are plotted in blue-ish colors, poisoned runs in red-ish colors.

    Args:
        all_run_results: A list of dictionaries, where each dictionary represents
                         a completed ExperimentRun (must contain 'run_type'
                         and 'belief_tracking_data').
        belief_query: The query string used for the belief task (for plot title).
        poison_step_indices: A list of step indices where poisoning was applied (for vlines).
        use_seaborn: If True, attempts to use Seaborn styling.

    Returns:
        matplotlib.figure.Figure: The generated plot figure object, or None if
                                  no valid data could be plotted.
    """
    # --- Plot Style Setup ---
    if use_seaborn:
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            # print("Using Seaborn style for plots.") # Quieter
        except ImportError:
            plt.style.use('seaborn-v0_8-whitegrid')
            # print("Seaborn not found, using fallback style.") # Quieter
    else:
         plt.style.use('default')
         # print("Using default Matplotlib style.") # Quieter


    # print("\n--- Starting Belief Analysis Visualization for Multiple Runs ---") # Quieter

    # --- Process Data for all runs ---
    processed_runs = []
    max_step_overall = 0
    num_clean = 0
    num_poisoned = 0

    if not all_run_results: # Handle empty input list
         print("\nERROR: No run results provided for plotting. Returning None.")
         return None

    for i, run_data in enumerate(all_run_results):
        run_type = run_data.get("run_type", "unknown")
        run_id = run_data.get("experiment_id", f"run_{i}")
        belief_data = run_data.get("belief_tracking_data")

        # print(f"Processing Run: {run_id} (Type: {run_type})...") # Quieter
        df_run = process_belief_data(belief_data, run_id)

        if not df_run.empty:
            processed_runs.append({"type": run_type, "id": run_id, "df": df_run})
            # Update overall max step only if 'step' column is not empty
            if 'step' in df_run.columns and not df_run['step'].empty:
                max_step_overall = max(max_step_overall, df_run['step'].max())
            if run_type == 'clean':
                num_clean += 1
            elif run_type == 'poisoned':
                num_poisoned +=1
        # else: # Quieter
            # print(f"-> No valid belief data found for {run_id}.")

    if not processed_runs:
        print("\nERROR: No valid data available from any run to generate plots. Returning None.")
        return None # Indicate no plot could be generated

    print(f"\nProcessed {len(processed_runs)} runs ({num_clean} clean, {num_poisoned} poisoned) for plotting.")

    # --- Visualization ---
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True) # Adjusted figure size
    marker_size = 90  # Reduced marker size from 120
    line_alpha = 0.6  # Adjusted transparency for lines
    marker_alpha = 0.7 # Adjusted transparency for markers

    # Define lighter colors
    color_clean = 'skyblue'
    color_poisoned = 'lightcoral'

    # Flags to ensure legends are added only once per type
    legend_flags = {
        "clean_line": False, "poisoned_line": False,
        "clean_marker_true": False, "clean_marker_false": False,
        "poisoned_marker_true": False, "poisoned_marker_false": False,
        "vline": False
    }

    # --- Plot 1: Belief Score (P(True)) ---
    ax[0].set_title(f'Belief Score (P(True)) for Query: "{belief_query}"\n({num_clean} Clean, {num_poisoned} Poisoned Runs)')
    ax[0].set_ylabel("Belief Score (P(True))")
    ax[0].set_ylim(-0.05, 1.05)

    for run in processed_runs:
        df = run["df"]
        run_type = run["type"]
        color = color_clean if run_type == "clean" else color_poisoned
        linestyle = '-' if run_type == "clean" else '--'
        line_label_key = f"{run_type}_line"
        marker_true_label_key = f"{run_type}_marker_true"
        marker_false_label_key = f"{run_type}_marker_false"

        # Plot Line (Belief Score) - ensure data exists and handle NaNs
        belief_data_clean = df[['step', 'belief_score']].dropna()
        if not belief_data_clean.empty:
            label_line = f"{run_type.capitalize()} Run (Belief)" if not legend_flags[line_label_key] else "_nolegend_"
            ax[0].plot(belief_data_clean['step'], belief_data_clean['belief_score'], marker='o', markersize=4, linestyle=linestyle,
                       label=label_line, color=color, alpha=line_alpha, zorder=2)
            if label_line != "_nolegend_": legend_flags[line_label_key] = True

        # Plot Markers (Belief Score)
        df_true = df[df['belief_numeric'] == 1].dropna(subset=['belief_score'])
        df_false = df[df['belief_numeric'] == 0].dropna(subset=['belief_score'])

        if not df_true.empty:
            label_marker_true = f'{run_type.capitalize()}: Gen="True"' if not legend_flags[marker_true_label_key] else "_nolegend_"
            ax[0].scatter(df_true['step'], df_true['belief_score'],
                         marker='P', color=color, s=marker_size, label=label_marker_true, zorder=3,
                         alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_true != "_nolegend_": legend_flags[marker_true_label_key] = True

        if not df_false.empty:
            label_marker_false = f'{run_type.capitalize()}: Gen="False"' if not legend_flags[marker_false_label_key] else "_nolegend_"
            ax[0].scatter(df_false['step'], df_false['belief_score'],
                         marker='X', color=color, s=marker_size, label=label_marker_false, zorder=3,
                         alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_false != "_nolegend_": legend_flags[marker_false_label_key] = True


    # Add vertical lines for poison steps
    if poison_step_indices:
        sorted_indices = sorted(list(set(poison_step_indices)))
        label_vline = 'Poison Step' # Consolidated label
        for ps in sorted_indices:
            current_label = label_vline if not legend_flags["vline"] else "_nolegend_"
            ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5,
                          label=current_label, zorder=1)
            if not legend_flags["vline"]: legend_flags["vline"] = True # Set flag after first line

    ax[0].legend(loc='best', fontsize='small')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # --- Plot 2: Generated Token Probability ---
    ax[1].set_title('Probability of Generated Answer (P(Generated Token))')
    ax[1].set_xlabel("Reasoning Step")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(-0.05, 1.05)

    # Reset legend flags for the second plot (except vline)
    legend_flags["clean_line"] = False; legend_flags["poisoned_line"] = False
    legend_flags["clean_marker_true"] = False; legend_flags["clean_marker_false"] = False
    legend_flags["poisoned_marker_true"] = False; legend_flags["poisoned_marker_false"] = False
    legend_flags["vline"] = False # Reset vline for second plot legend

    for run in processed_runs:
        df = run["df"]
        run_type = run["type"]
        color = color_clean if run_type == "clean" else color_poisoned
        linestyle = '-' if run_type == "clean" else '--'
        line_label_key = f"{run_type}_line"
        marker_true_label_key = f"{run_type}_marker_true"
        marker_false_label_key = f"{run_type}_marker_false"

        # Plot Line (Generated Probability) - ensure data exists and handle NaNs
        prob_gen_data_clean = df[['step', 'prob_generated']].dropna()
        if not prob_gen_data_clean.empty:
             label_line = f"{run_type.capitalize()} Run (P(Gen))" if not legend_flags[line_label_key] else "_nolegend_"
             ax[1].plot(prob_gen_data_clean['step'], prob_gen_data_clean['prob_generated'], marker='o', markersize=4, linestyle=linestyle,
                       label=label_line, color=color, alpha=line_alpha, zorder=2)
             if label_line != "_nolegend_": legend_flags[line_label_key] = True

        # Plot Markers (Generated Probability)
        df_true = df[df['belief_numeric'] == 1].dropna(subset=['prob_generated'])
        df_false = df[df['belief_numeric'] == 0].dropna(subset=['prob_generated'])

        if not df_true.empty:
            label_marker_true = f'{run_type.capitalize()}: Gen="True"' if not legend_flags[marker_true_label_key] else "_nolegend_"
            ax[1].scatter(df_true['step'], df_true['prob_generated'],
                         marker='P', color=color, s=marker_size, label=label_marker_true, zorder=3,
                         alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_true != "_nolegend_": legend_flags[marker_true_label_key] = True

        if not df_false.empty:
            label_marker_false = f'{run_type.capitalize()}: Gen="False"' if not legend_flags[marker_false_label_key] else "_nolegend_"
            ax[1].scatter(df_false['step'], df_false['prob_generated'],
                         marker='X', color=color, s=marker_size, label=label_marker_false, zorder=3,
                         alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_false != "_nolegend_": legend_flags[marker_false_label_key] = True


    # Add vertical lines for poison steps
    if poison_step_indices:
         sorted_indices = sorted(list(set(poison_step_indices)))
         label_vline = 'Poison Step' # Consolidated label
         for ps in sorted_indices:
            current_label = label_vline if not legend_flags["vline"] else "_nolegend_"
            ax[1].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5,
                          label=current_label, zorder=1)
            if not legend_flags["vline"]: legend_flags["vline"] = True # Set flag after first line

    ax[1].legend(loc='best', fontsize='small')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- General Plot Adjustments ---
    if max_step_overall >= 0: # Check >= 0 to handle single step case
        # Adjust x-ticks for readability
        tick_step = max(1, int(np.ceil((max_step_overall + 1) / 10))) # Aim for ~10 ticks based on max_step+1 range
        # Ensure ticks cover the range, including 0 and max_step_overall
        ticks = np.arange(0, max_step_overall + tick_step, tick_step)
        ticks = ticks[ticks <= max_step_overall] # Remove ticks beyond max step
        if 0 not in ticks: ticks = np.insert(ticks, 0, 0) # Ensure 0 is included
        if max_step_overall not in ticks and max_step_overall > 0: ticks = np.append(ticks, max_step_overall) # Ensure max step included

        plt.xticks(ticks=np.unique(ticks)) # Use unique ticks in case 0 or max was added twice
        plt.xlim(left=-0.5, right=max_step_overall + 0.5) # Add padding
    else: # Handle case with no steps (shouldn't happen if processed_runs is not empty)
         plt.xticks([0])
         plt.xlim(left=-0.5, right=0.5)


    fig.tight_layout(pad=1.5) # Adjust layout slightly

    # print("--- Visualization Complete ---") # Quieter

    return fig # Return the figure object