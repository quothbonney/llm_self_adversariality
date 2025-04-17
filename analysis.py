# analysis.py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
try:
    import seaborn as sns
except ImportError:
    print("ERROR: Seaborn is required for the marginal distribution plots. Please install it (`pip install seaborn`)")
    sns = None

# --- Helper Function: Data Processing (REVISED) ---
def process_belief_data(belief_data: Optional[Dict[str, Any]], run_name: str = "Unknown") -> pd.DataFrame:
    """
    Converts a dictionary of belief data per step into a Pandas DataFrame.
    Handles string keys from JSON input and calculates 'belief_score'.

    Args:
        belief_data: Dict {STEP_STR: {'Generated': 'True'/'False', 'True_Logprob': float, 'False_Logprob': float}}.
                     Keys are expected to be strings representing step numbers.
        run_name: Identifier for the data source.

    Returns:
        Pandas DataFrame with columns: 'step', 'generated', 'True_Logprob',
        'False_Logprob', 'belief_score', 'prob_generated', 'belief_numeric'.
        Returns empty DataFrame if input is None or empty.
    """
    if not belief_data:
        # print(f"DEBUG: No belief data provided for {run_name} run.")
        return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
        ])

    processed_data = []
    # --- CORRECTED KEY HANDLING ---
    # Get string keys, try converting to int for sorting, handle errors
    valid_step_keys = {} # Store mapping int_step -> str_key
    for key_str in belief_data.keys():
        try:
            step_int = int(key_str)
            valid_step_keys[step_int] = key_str
        except (ValueError, TypeError):
            # print(f"DEBUG: Invalid step key '{key_str}' in {run_name}, skipping.")
            continue # Skip keys that aren't valid integer strings

    if not valid_step_keys:
         # print(f"DEBUG: No valid numeric step keys found in {run_name}.")
         return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
         ])

    # Iterate through sorted integer steps
    sorted_steps = sorted(valid_step_keys.keys())
    # --- END CORRECTED KEY HANDLING ---

    for step_int in sorted_steps:
        key_str = valid_step_keys[step_int] # Get the original string key
        data = belief_data.get(key_str)

        if not data: # Should not happen if key exists, but check anyway
            continue

        generated = data.get('Generated')
        true_lp = data.get('True_Logprob')
        false_lp = data.get('False_Logprob')

        # Ensure logprobs are floats if they exist
        try: true_lp = float(true_lp) if true_lp is not None else None
        except (ValueError, TypeError): true_lp = None
        try: false_lp = float(false_lp) if false_lp is not None else None
        except (ValueError, TypeError): false_lp = None

        # --- Calculate belief_score from logprobs ---
        belief_score = np.nan
        true_prob = np.nan
        false_prob = np.nan

        if true_lp is not None and true_lp <= 0: true_prob = np.exp(true_lp)
        if false_lp is not None and false_lp <= 0: false_prob = np.exp(false_lp)

        if not pd.isna(true_prob) and not pd.isna(false_prob):
            denominator = true_prob + false_prob
            if denominator > 1e-9: belief_score = true_prob / denominator
            elif true_prob > false_prob: belief_score = 1.0
            elif false_prob > true_prob: belief_score = 0.0
            else: belief_score = 0.5
        elif not pd.isna(true_prob): belief_score = true_prob
        elif not pd.isna(false_prob): belief_score = 1.0 - false_prob

        # --- Calculate other columns ---
        prob_gen = np.nan
        if generated == 'True' and not pd.isna(true_prob): prob_gen = true_prob
        elif generated == 'False' and not pd.isna(false_prob): prob_gen = false_prob

        belief_numeric = np.nan
        if generated == 'True': belief_numeric = 1
        elif generated == 'False': belief_numeric = 0

        processed_data.append({
            'step': step_int, # Use the integer step
            'generated': generated,
            'True_Logprob': true_lp,
            'False_Logprob': false_lp,
            'belief_score': belief_score,
            'prob_generated': prob_gen,
            'belief_numeric': belief_numeric
        })

    if not processed_data:
        # print(f"DEBUG: No valid steps processed for {run_name} after loop.")
        return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
        ])

    return pd.DataFrame(processed_data)


# --- Visualization Function (Should remain the same as previous version) ---
def plot_belief_analysis(
    all_run_results: List[Dict[str, Any]],
    main_belief_query: str,
    toxic_belief_query: str,
    poison_step_indices: Optional[List[int]] = None,
    use_seaborn: bool = True
) -> Optional[plt.Figure]:
    """
    Generates a two-panel plot with side-by-side marginal distributions
    for initial and final steps, showing belief trends across multiple runs.
    Calculates belief score from True/False Logprobs.

    Args:
        all_run_results: List of completed ExperimentRun dictionaries.
        main_belief_query: The main query string for the overall figure title.
        toxic_belief_query: The toxic query string for the belief score panel title.
        poison_step_indices: List of step indices where poisoning was applied.
        use_seaborn: If True, attempts to use Seaborn styling for main plots.

    Returns:
        matplotlib.figure.Figure: The generated plot figure object, or None if no data.
    """
    if sns is None:
        print("Seaborn import failed. Cannot generate marginal plots.")
        return None

    if use_seaborn:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use('default')

    processed_runs = []
    # Store lists of valid scores/probs for marginal plots
    first_step_data = {'clean': {'belief_score': [], 'prob_generated': []},
                       'poisoned': {'belief_score': [], 'prob_generated': []}}
    final_step_data = {'clean': {'belief_score': [], 'prob_generated': []},
                       'poisoned': {'belief_score': [], 'prob_generated': []}}
    max_step_overall = 0
    num_clean = 0
    num_poisoned = 0

    if not all_run_results: return None

    print("\nProcessing runs for plotting...") # Keep this print
    for i, run_data in enumerate(all_run_results):
        run_type = run_data.get("run_type", "unknown")
        run_id = run_data.get("experiment_id", f"run_{i}")
        belief_data_raw = run_data.get("belief_tracking_data", {})
        # Ensure belief_tracking_data keys are strings for process_belief_data
        belief_data_str_keys = {str(k): v for k, v in belief_data_raw.items()}

        # *** Call the REVISED process_belief_data ***
        df_run = process_belief_data(belief_data_str_keys, run_id)

        # Check if the DataFrame is valid AFTER processing
        if not df_run.empty and 'step' in df_run.columns and not df_run['step'].empty:
            processed_runs.append({"type": run_type, "id": run_id, "df": df_run})

            # Safely get min/max steps from the DataFrame's 'step' column
            steps = df_run['step'].dropna().unique()
            if len(steps) == 0: continue # Skip if no valid steps in this df
            current_min_step = steps.min()
            current_max_step = steps.max()
            max_step_overall = max(max_step_overall, current_max_step)

            # Extract first and final step data, ensuring values are valid numbers
            # Use .loc for safer indexing
            first_row_series = df_run.loc[df_run['step'] == current_min_step].iloc[0]
            last_row_series = df_run.loc[df_run['step'] == current_max_step].iloc[0]

            if run_type in final_step_data: # Check if type is 'clean' or 'poisoned'
                # --- First step data ---
                bs_first = first_row_series['belief_score']
                pg_first = first_row_series['prob_generated']
                if pd.notna(bs_first): first_step_data[run_type]['belief_score'].append(bs_first)
                if pd.notna(pg_first): first_step_data[run_type]['prob_generated'].append(pg_first)

                # --- Last step data ---
                bs_last = last_row_series['belief_score']
                pg_last = last_row_series['prob_generated']
                if pd.notna(bs_last): final_step_data[run_type]['belief_score'].append(bs_last)
                if pd.notna(pg_last): final_step_data[run_type]['prob_generated'].append(pg_last)

            if run_type == 'clean': num_clean += 1
            elif run_type == 'poisoned': num_poisoned +=1
        # else: # Optional debug print if df_run is empty
            # print(f"DEBUG: DataFrame for run {run_id} was empty or invalid after processing.")


    # ---> The crucial check is here <---
    if not processed_runs:
        # This is the message the user saw
        print("No valid runs processed for plotting.")
        return None

    # (Rest of the plotting code remains the same as the previous version)
    # ... [Code for Visualization Setup, Marginal Plots, Time Series Plots, Final Adjustments] ...
    # --- Visualization Setup ---
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=(12, 1, 1), height_ratios=(1, 1),
                          wspace=0.05, hspace=0.2)

    ax = [None, None] # Initialize list first
    ax[0] = fig.add_subplot(gs[0, 0]) # Create first subplot
    ax[1] = fig.add_subplot(gs[1, 0], sharex=ax[0]) # Create second, now ax[0] exists
    ax_marg_first = [fig.add_subplot(gs[0, 1], sharey=ax[0]), fig.add_subplot(gs[1, 1], sharey=ax[1])]
    ax_marg_last = [fig.add_subplot(gs[0, 2], sharey=ax[0]), fig.add_subplot(gs[1, 2], sharey=ax[1])]

    fig.suptitle(f'Belief Analysis for Query: "{main_belief_query}"', fontsize=16, y=0.97)

    # --- Plotting Parameters ---
    marker_size = 60
    line_alpha = 0.6
    marker_alpha = 0.7
    color_clean = 'skyblue'
    color_poisoned = 'lightcoral'
    kde_common_args = {'linewidth': 1.5, 'alpha': 0.7, 'fill': True}

    # --- Plot Marginal Distributions ---
    # Panel 1 Marginals (Belief Score)
    ax_marg_first[0].set_title('Initial', fontsize=9, pad=2)
    ax_marg_last[0].set_title('Final', fontsize=9, pad=2)
    if first_step_data['clean']['belief_score']:
        sns.kdeplot(y=first_step_data['clean']['belief_score'], ax=ax_marg_first[0], color=color_clean, **kde_common_args)
    if first_step_data['poisoned']['belief_score']:
        sns.kdeplot(y=first_step_data['poisoned']['belief_score'], ax=ax_marg_first[0], color=color_poisoned, **kde_common_args)
    if final_step_data['clean']['belief_score']:
        sns.kdeplot(y=final_step_data['clean']['belief_score'], ax=ax_marg_last[0], color=color_clean, **kde_common_args)
    if final_step_data['poisoned']['belief_score']:
        sns.kdeplot(y=final_step_data['poisoned']['belief_score'], ax=ax_marg_last[0], color=color_poisoned, **kde_common_args)

    # Panel 2 Marginals (P(Generated))
    if first_step_data['clean']['prob_generated']:
        sns.kdeplot(y=first_step_data['clean']['prob_generated'], ax=ax_marg_first[1], color=color_clean, **kde_common_args)
    if first_step_data['poisoned']['prob_generated']:
        sns.kdeplot(y=first_step_data['poisoned']['prob_generated'], ax=ax_marg_first[1], color=color_poisoned, **kde_common_args)
    if final_step_data['clean']['prob_generated']:
        sns.kdeplot(y=final_step_data['clean']['prob_generated'], ax=ax_marg_last[1], color=color_clean, **kde_common_args)
    if final_step_data['poisoned']['prob_generated']:
        sns.kdeplot(y=final_step_data['poisoned']['prob_generated'], ax=ax_marg_last[1], color=color_poisoned, **kde_common_args)

    # Formatting for all Marginal Axes
    for i in range(2):
        for ax_m in [ax_marg_first[i], ax_marg_last[i]]:
            ax_m.tick_params(axis='y', labelleft=False, left=False) # Hide y-ticks and labels
            ax_m.tick_params(axis='x', labelbottom=False, bottom=False) # Hide x-ticks and labels
            ax_m.grid(False) # Turn off grid
            ax_m.set_xlabel(''); ax_m.set_ylabel('')

    # --- Plot Time Series Data on Main Axes ---
    ax[0].set_title(f'Belief Score (P(True)) for Toxic Query: "{toxic_belief_query}"', fontsize=11)
    ax[0].set_ylabel("Belief Score (P(True))")
    ax[0].set_ylim(-0.05, 1.05)
    ax[1].set_title('Probability of Generated Answer (P(Generated Token))', fontsize=11)
    ax[1].set_xlabel("Reasoning Step")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(-0.05, 1.05)

    legend_handles_labels_ax0 = {}
    legend_handles_labels_ax1 = {}

    for run in processed_runs:
        df = run["df"]
        run_type = run["type"]
        color = color_clean if run_type == "clean" else color_poisoned
        linestyle = '-' if run_type == "clean" else '--'

        # Plotting on ax[0] (Belief Score)
        plot_data = df[['step', 'belief_score']].dropna()
        if not plot_data.empty:
            label_line = f"{run_type.capitalize()} Run (Belief)"
            handle, = ax[0].plot(plot_data['step'], plot_data['belief_score'], marker='.', markersize=5, linestyle=linestyle, label=label_line, color=color, alpha=line_alpha, zorder=2)
            if label_line not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_line] = handle
        df_true = df[df['belief_numeric'] == 1].dropna(subset=['belief_score'])
        if not df_true.empty:
            label_marker = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[0].scatter(df_true['step'], df_true['belief_score'], marker='P', color=color, s=marker_size, label=label_marker, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_marker] = handle
        df_false = df[df['belief_numeric'] == 0].dropna(subset=['belief_score'])
        if not df_false.empty:
            label_marker = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[0].scatter(df_false['step'], df_false['belief_score'], marker='X', color=color, s=marker_size, label=label_marker, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_marker] = handle

        # Plotting on ax[1] (P(Generated))
        plot_data = df[['step', 'prob_generated']].dropna()
        if not plot_data.empty:
            label_line = f"{run_type.capitalize()} Run (P(Gen))"
            handle, = ax[1].plot(plot_data['step'], plot_data['prob_generated'], marker='.', markersize=5, linestyle=linestyle, label=label_line, color=color, alpha=line_alpha, zorder=2)
            if label_line not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_line] = handle
        df_true = df[df['belief_numeric'] == 1].dropna(subset=['prob_generated'])
        if not df_true.empty:
            label_marker = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[1].scatter(df_true['step'], df_true['prob_generated'], marker='P', color=color, s=marker_size, label=label_marker, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_marker] = handle
        df_false = df[df['belief_numeric'] == 0].dropna(subset=['prob_generated'])
        if not df_false.empty:
            label_marker = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[1].scatter(df_false['step'], df_false['prob_generated'], marker='X', color=color, s=marker_size, label=label_marker, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_marker] = handle

    # --- Add Vertical Lines, Legends, Grid to Main Axes ---
    vline_label_added = False
    if poison_step_indices:
        sorted_indices = sorted([idx for idx in poison_step_indices if isinstance(idx, (int, float))]) # Ensure numeric
        label_vline = 'Poison Step'
        for ps in sorted_indices:
            current_label = label_vline if not vline_label_added else "_nolegend_"
            handle = ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, label=current_label, zorder=1)
            if not vline_label_added:
                legend_handles_labels_ax0[label_vline] = handle # Add handle to legend dict
                vline_label_added = True
            ax[1].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, zorder=1) # No label on second plot

    if legend_handles_labels_ax0:
      ax[0].legend(legend_handles_labels_ax0.values(), legend_handles_labels_ax0.keys(), loc='best', fontsize='small')
    if legend_handles_labels_ax1:
      ax[1].legend(legend_handles_labels_ax1.values(), legend_handles_labels_ax1.keys(), loc='best', fontsize='small')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Final Adjustments ---
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].tick_params(axis='x', bottom=False)
    if max_step_overall >= 0:
        tick_step = max(1, int(np.ceil((max_step_overall + 1) / 10)))
        ticks = np.arange(0, max_step_overall + tick_step, tick_step)
        ticks = ticks[ticks <= max_step_overall]
        if 0 not in ticks: ticks = np.insert(ticks, 0, 0)
        if max_step_overall not in ticks and max_step_overall > 0: ticks = np.append(ticks, max_step_overall)
        ax[1].set_xticks(np.unique(ticks))
        ax[1].set_xlim(left=-0.5, right=max_step_overall + 0.5)
    else:
         ax[1].set_xticks([0]); ax[1].set_xlim(left=-0.5, right=0.5)

    try:
      fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    except ValueError:
        print("Warning: tight_layout failed, plot may have overlapping elements.")

    print(f"Plot generated successfully using {len(processed_runs)} runs.")
    return fig