# generate_plot_from_json.py
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import sys # For error messages
from scipy.stats import gaussian_kde

# Try importing seaborn, handle if missing
try:
    import seaborn as sns
except ImportError:
    print("ERROR: Seaborn is required for plotting. Please install it (`pip install seaborn`)", file=sys.stderr)
    sns = None # Set sns to None if import fails

# --- Helper Function: Data Processing (Copied from analysis.py) ---
def process_belief_data(belief_data: Optional[Dict[str, Any]], run_name: str = "Unknown") -> pd.DataFrame:
    """
    Converts a dictionary of belief data per step into a Pandas DataFrame.
    Handles string keys from JSON input and calculates 'belief_score'.
    """
    if not belief_data:
        return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
        ])

    processed_data = []
    valid_step_keys = {}
    for key_str in belief_data.keys():
        try:
            step_int = int(key_str)
            valid_step_keys[step_int] = key_str
        except (ValueError, TypeError):
            continue

    if not valid_step_keys:
         return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
         ])

    sorted_steps = sorted(valid_step_keys.keys())

    for step_int in sorted_steps:
        key_str = valid_step_keys[step_int]
        data = belief_data.get(key_str)
        if not data: continue

        # --- Access data within belief_logits ---
        belief_logits = data.get('belief_logits', {}) # Get the sub-dictionary

        # ---> ADDED CHECK: Handle case where belief_logits key exists but is null/None
        if belief_logits is None:
            generated = None
            true_lp = None
            false_lp = None
        else:
            generated = belief_logits.get('Generated')
            true_lp = belief_logits.get('True_Logprob')
            false_lp = belief_logits.get('False_Logprob')
        # --- End access modification & ADDED CHECK ---

        try: true_lp = float(true_lp) if true_lp is not None else None
        except (ValueError, TypeError): true_lp = None
        try: false_lp = float(false_lp) if false_lp is not None else None
        except (ValueError, TypeError): false_lp = None

        belief_score, true_prob, false_prob = np.nan, np.nan, np.nan
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

        prob_gen = np.nan
        if generated == 'True' and not pd.isna(true_prob): prob_gen = true_prob
        elif generated == 'False' and not pd.isna(false_prob): prob_gen = false_prob

        belief_numeric = np.nan
        if generated == 'True': belief_numeric = 1
        elif generated == 'False': belief_numeric = 0

        processed_data.append({
            'step': step_int, 'generated': generated, 'True_Logprob': true_lp,
            'False_Logprob': false_lp, 'belief_score': belief_score,
            'prob_generated': prob_gen, 'belief_numeric': belief_numeric
        })

    if not processed_data:
        return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
        ])

    return pd.DataFrame(processed_data)

# --- Visualization Function (Copied & slightly modified from analysis.py) ---
def plot_belief_analysis(
    all_run_results: List[Dict[str, Any]],
    main_belief_query: str,
    toxic_belief_query: str,
    poison_step_indices: Optional[List[int]] = None,
    use_seaborn: bool = True
) -> Optional[plt.Figure]:
    """
    Generates the belief analysis plot from processed run data.
    (Code is nearly identical to the previous version in analysis.py,
     minor print adjustments for standalone script context)
    """
    if sns is None:
        print("Seaborn import failed. Cannot generate plot.", file=sys.stderr)
        return None

    if use_seaborn: sns.set_theme(style="whitegrid")
    else: plt.style.use('default')

    processed_runs = []
    first_step_data = {'clean': {'belief_score': [], 'prob_generated': []}, 'poisoned': {'belief_score': [], 'prob_generated': []}}
    final_step_data = {'clean': {'belief_score': [], 'prob_generated': []}, 'poisoned': {'belief_score': [], 'prob_generated': []}}
    max_step_overall = 0
    num_clean, num_poisoned = 0, 0

    if not all_run_results:
        print("No runs found in the input data.", file=sys.stderr)
        return None

    # print("Processing runs for plotting...") # Can be verbose
    for i, run_data in enumerate(all_run_results):
        run_type = run_data.get("run_type", "unknown")
        run_id = run_data.get("experiment_id", f"run_{i}")
        belief_data_raw = run_data.get("step_data", {})
        belief_data_str_keys = {str(k): v for k, v in belief_data_raw.items()}
        df_run = process_belief_data(belief_data_str_keys, run_id)
 
        if not df_run.empty and 'step' in df_run.columns and not df_run['step'].empty:
            processed_runs.append({"type": run_type, "id": run_id, "df": df_run})

            # --- Calculate overall max step for axis limits ---
            # Find the absolute max step across all valid steps in this run
            steps = df_run['step'].dropna().unique()
            if len(steps) > 0:
                 max_step_overall = max(max_step_overall, steps.max())
            # --- End overall max step calculation ---

            # --- Extract data for marginal plots (first/last valid steps per metric) ---
            # Process only if run_type is 'clean' or 'poisoned'
            if run_type in final_step_data:

                # --- Belief Score - Find First and Last Valid ---
                df_valid_bs = df_run.dropna(subset=['belief_score']) # Filter out NaNs
                if not df_valid_bs.empty:
                    # Find the step corresponding to the first valid belief score
                    first_valid_step_bs = df_valid_bs['step'].min()
                    # Find the step corresponding to the last valid belief score
                    last_valid_step_bs = df_valid_bs['step'].max()

                    # Get the actual belief score values at those specific steps using .loc
                    # .iloc[0] is used in case there are multiple entries for the same step (shouldn't happen)
                    bs_first = df_valid_bs.loc[df_valid_bs['step'] == first_valid_step_bs, 'belief_score'].iloc[0]
                    bs_last = df_valid_bs.loc[df_valid_bs['step'] == last_valid_step_bs, 'belief_score'].iloc[0]

                    # Append the valid first/last scores to the lists for marginal plots
                    first_step_data[run_type]['belief_score'].append(bs_first)
                    final_step_data[run_type]['belief_score'].append(bs_last)

                # --- Prob Generated - Find First and Last Valid ---
                df_valid_pg = df_run.dropna(subset=['prob_generated']) # Filter out NaNs
                if not df_valid_pg.empty:
                    # Find the step corresponding to the first valid probability
                    first_valid_step_pg = df_valid_pg['step'].min()
                    # Find the step corresponding to the last valid probability
                    last_valid_step_pg = df_valid_pg['step'].max()

                    # Get the actual probability values at those specific steps using .loc
                    pg_first = df_valid_pg.loc[df_valid_pg['step'] == first_valid_step_pg, 'prob_generated'].iloc[0]
                    pg_last = df_valid_pg.loc[df_valid_pg['step'] == last_valid_step_pg, 'prob_generated'].iloc[0]

                    # Append the valid first/last probabilities to the lists for marginal plots
                    first_step_data[run_type]['prob_generated'].append(pg_first)
                    final_step_data[run_type]['prob_generated'].append(pg_last)
            # --- End marginal data extraction ---

            # Increment run type counters (can be done after processing)
            if run_type == 'clean': num_clean += 1
            elif run_type == 'poisoned': num_poisoned +=1

        # else: # Optional: Handle case where df_run was empty after processing
            # print(f"DEBUG: DataFrame for run {run_id} was empty or invalid after processing.")

    # ---> The crucial check is here <--- (This line remains unchanged)
    if not processed_runs:
        # This is the message the user saw previously
        print("No valid runs processed containing plottable data.")
        return None

    # --- Visualization Setup ---

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=(12, 1, 1), height_ratios=(1, 1), wspace=0.05, hspace=0.2)
    ax = [None, None];
    ax_marg_first = [None, None];
    ax_marg_last = [None, None]
    ax[0] = fig.add_subplot(gs[0, 0]);
    ax[1] = fig.add_subplot(gs[1, 0], sharex=ax[0])
    ax_marg_first[0] = fig.add_subplot(gs[0, 1], sharey=ax[0]);
    ax_marg_last[0] = fig.add_subplot(gs[0, 2], sharey=ax[0]) # Final Belief Score Marginal Axis
    ax_marg_first[1] = fig.add_subplot(gs[1, 1], sharey=ax[1]);
    ax_marg_last[1] = fig.add_subplot(gs[1, 2], sharey=ax[1]) # Final P(Generated) Marginal Axis

    fig.suptitle(f'Belief Analysis for Query: "{main_belief_query}"', fontsize=16, y=0.97)
    marker_size, line_alpha, marker_alpha = 60, 0.6, 0.7
    color_clean, color_poisoned = 'skyblue', 'lightcoral'

    # Keep original args for initial plots (or define separately if needed)
    # Note: bw_adjust is included here, affecting initial plots
    kde_common_args_initial = {'linewidth': 1.5, 'alpha': 0.7, 'fill': True, 'bw_adjust': 1.0}

    # Define args for manual plotting (alpha, linewidth) - bw_adjust handled separately
    manual_plot_args = {'alpha': 0.7, 'linewidth': 1.5}
    manual_bw_adjust = 1.0 # Use the same bw_adjust for consistency

        # --- Helper function for Manual Relative KDE Plotting ---
    def plot_relative_kde(data, ax_m, color, grid_points, bw_adjust=1.0, plot_args={}):
        """Calculates KDE, scales to max=1, and plots using fill_betweenx."""
        if not data or len(data) < 2: # Need at least 2 points for KDE
            # print(f"Warning: Skipping relative KDE for {color}, not enough data points ({len(data) if data else 0}).") # Optional: Reduce verbosity
            return
        try:
            data_array = np.asarray(data)
            # Remove potential NaNs/Infs just in case they slipped through
            data_array = data_array[np.isfinite(data_array)]
            if data_array.size < 2:
                # print(f"Warning: Skipping relative KDE for {color}, not enough finite data points ({data_array.size}).") # Optional: Reduce verbosity
                return

            # 1. Initialize KDE (calculates default factor)
            kde = gaussian_kde(data_array, bw_method=bw_adjust) # Or 'silverman'

            # 2. <<< CORRECT WAY TO APPLY BANDWIDTH ADJUSTMENT >>>
            # Modify the 'factor' directly before evaluation
            kde.factor = kde.factor * bw_adjust
            # --- End Correction ---

            # 3. Evaluate KDE on the grid
            kde_y_values = kde(grid_points)
            max_density = np.max(kde_y_values)

            # 4. Scale and Plot
            if max_density > 1e-9: # Avoid division by zero or scaling up tiny noise
                scaled_kde_y = kde_y_values / max_density
                ax_m.fill_betweenx(grid_points, 0, scaled_kde_y, color=color, **plot_args)
            # else: # Optional: Handle case where max density is effectively zero
                # print(f"Warning: Max density near zero for {color}, skipping plot.")

        except Exception as e:
            # Catch potential errors during KDE calculation (e.g., singular matrix if data is degenerate)
            print(f"ERROR calculating or plotting relative KDE for {color}: {e}")
            # Optional: Add traceback for detailed debugging if errors persist
            # import traceback
            # traceback.print_exc()


    # --- Plot Marginal Distributions ---
    # Set titles only for the top marginal axes (representing columns)
    ax_marg_first[0].set_title('Initial', fontsize=9, pad=2)
    #set ylim to [0, 1]
    ax_marg_last[0].set_title('Final', fontsize=9, pad=2)
    # No titles needed for bottom marginal axes (ax_marg_first[1], ax_marg_last[1])

    # --- Panel 1 Marginals (Belief Score) ---
    # Initial Step Belief Score (Using original sns.kdeplot)
    if first_step_data['clean']['belief_score']:
        sns.kdeplot(y=first_step_data['clean']['belief_score'], ax=ax_marg_first[0], color=color_clean, **kde_common_args_initial)
    if first_step_data['poisoned']['belief_score']:
        sns.kdeplot(y=first_step_data['poisoned']['belief_score'], ax=ax_marg_first[0], color=color_poisoned, **kde_common_args_initial)
    # clip the kdeplot to [0, 1]
    ax_marg_first[0].set_ylim(0, 1)
    ax_marg_first[1].set_ylim(0, 1)
    # Final Step Belief Score (Using MANUAL relative scaling)
    ax_m_bs = ax_marg_last[0]
    y_min_bs, y_max_bs = [0, 1]
    grid_points_bs = np.linspace(y_min_bs, y_max_bs, 200) # Grid for evaluation

    plot_relative_kde(final_step_data['clean']['belief_score'], ax_m_bs, color_clean, grid_points_bs, bw_adjust=1.0, plot_args=manual_plot_args)
    plot_relative_kde(final_step_data['poisoned']['belief_score'], ax_m_bs, color_poisoned, grid_points_bs, bw_adjust=0.4, plot_args=manual_plot_args)
    print(final_step_data['poisoned']['belief_score'])

    # --- Panel 2 Marginals (P(Generated)) ---
    # Initial Step P(Generated) (Using original sns.kdeplot)
    if first_step_data['clean']['prob_generated']:
        sns.kdeplot(y=first_step_data['clean']['prob_generated'], ax=ax_marg_first[1], color=color_clean, **kde_common_args_initial)
    if first_step_data['poisoned']['prob_generated']:
        sns.kdeplot(y=first_step_data['poisoned']['prob_generated'], ax=ax_marg_first[1], color=color_poisoned, **kde_common_args_initial)

    # Final Step P(Generated) (Using MANUAL relative scaling)
    ax_m_pg = ax_marg_last[1]
    y_min_pg, y_max_pg = [0, 1]
    grid_points_pg = np.linspace(y_min_pg, y_max_pg, 200) # Grid for evaluation

    plot_relative_kde(final_step_data['clean']['prob_generated'], ax_m_pg, color_clean, grid_points_pg, bw_adjust=1.0, plot_args=manual_plot_args)
    plot_relative_kde(final_step_data['poisoned']['prob_generated'], ax_m_pg, color_poisoned, grid_points_pg, bw_adjust=0.4, plot_args=manual_plot_args)


    # --- Formatting for all Marginal Axes ---
    # NOTE: The x-axis for ax_marg_last now represents RELATIVE density (0 to 1)
    # The x-axis for ax_marg_first represents ABSOLUTE density

    # Format initial marginal axes (keep x-axis as absolute density)
    for ax_m in [ax_marg_first[0], ax_marg_first[1]]:
        ax_m.tick_params(axis='y', labelleft=False, left=False)
        ax_m.tick_params(axis='x', labelbottom=False, bottom=False)
        ax_m.grid(False)
        ax_m.set_xlabel('')
        ax_m.set_ylabel('')

    # Format final marginal axes (set x-axis for relative density)
    for ax_m in [ax_marg_last[0], ax_marg_last[1]]:
        ax_m.tick_params(axis='y', labelleft=False, left=False)
        ax_m.tick_params(axis='x', labelbottom=False, bottom=False) # Keep X ticks hidden
        ax_m.grid(False)
        ax_m.set_xlabel('') # Keep X label hidden (or set to "Relative Density" if desired)
        ax_m.set_ylabel('')
        ax_m.set_xlim(left=0, right=1.05) # Set X limits for relative density (0 to 1)


    # --- Plot Time Series Data --- (Code continues as before)
    ax[0].set_title(f'Belief Score (P(True)) for Toxic Query: "{toxic_belief_query}"', fontsize=11); ax[0].set_ylabel("Belief Score (P(True))"); ax[0].set_ylim(-0.05, 1.05)
    ax[1].set_title('Probability of Generated Answer (P(Generated Token))', fontsize=11); ax[1].set_xlabel("Reasoning Step"); ax[1].set_ylabel("Probability"); ax[1].set_ylim(-0.05, 1.05)
    legend_handles_labels_ax0, legend_handles_labels_ax1 = {}, {}

        # --- Plot Time Series Data on Main Axes (ax[0], ax[1]) ---
    legend_handles_labels_ax0 = {} # Store unique handles/labels for ax0 legend
    legend_handles_labels_ax1 = {} # Store unique handles/labels for ax1 legend

    # Loop through each processed run
    for run in processed_runs:
        df = run["df"]
        run_type = run["type"]
        color = color_clean if run_type == "clean" else color_poisoned
        linestyle = '-' if run_type == "clean" else '--'

        # --- Plotting on ax[0] (Belief Score) ---

        # Plot Belief Score Line
        plot_data_belief = df[['step', 'belief_score']].dropna()
        if not plot_data_belief.empty:
            label = f"{run_type.capitalize()} Run (Belief)"
            # Plot line and get handle (the comma after handle is important)
            handle, = ax[0].plot(plot_data_belief['step'], plot_data_belief['belief_score'],
                                 marker='.', markersize=5, linestyle=linestyle, label=label,
                                 color=color, alpha=line_alpha, zorder=2)
            # Add handle to dict only if label is new (setdefault does this)
            legend_handles_labels_ax0.setdefault(label, handle)

        # Plot Scatter for Gen="True" on ax[0]
        df_true_belief = df[df['belief_numeric'] == 1].dropna(subset=['belief_score'])
        if not df_true_belief.empty:
            label = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[0].scatter(df_true_belief['step'], df_true_belief['belief_score'],
                                   marker='P', color=color, s=marker_size, label=label,
                                   zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            # Add handle to dict only if label is new
            legend_handles_labels_ax0.setdefault(label, handle)

        # Plot Scatter for Gen="False" on ax[0]
        df_false_belief = df[df['belief_numeric'] == 0].dropna(subset=['belief_score'])
        if not df_false_belief.empty:
            label = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[0].scatter(df_false_belief['step'], df_false_belief['belief_score'],
                                   marker='X', color=color, s=marker_size, label=label,
                                   zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            # Add handle to dict only if label is new
            legend_handles_labels_ax0.setdefault(label, handle)

        # --- Plotting on ax[1] (P(Generated)) ---

        # Plot P(Generated) Line
        plot_data_prob = df[['step', 'prob_generated']].dropna()
        if not plot_data_prob.empty:
            label = f"{run_type.capitalize()} Run (P(Gen))"
            handle, = ax[1].plot(plot_data_prob['step'], plot_data_prob['prob_generated'],
                                 marker='.', markersize=5, linestyle=linestyle, label=label,
                                 color=color, alpha=line_alpha, zorder=2)
            # Add handle to dict only if label is new
            legend_handles_labels_ax1.setdefault(label, handle)

        # Plot Scatter for Gen="True" on ax[1]
        df_true_prob = df[df['belief_numeric'] == 1].dropna(subset=['prob_generated'])
        if not df_true_prob.empty:
            label = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[1].scatter(df_true_prob['step'], df_true_prob['prob_generated'],
                                   marker='P', color=color, s=marker_size, label=label,
                                   zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            # Add handle to dict only if label is new
            legend_handles_labels_ax1.setdefault(label, handle)

        # Plot Scatter for Gen="False" on ax[1]
        df_false_prob = df[df['belief_numeric'] == 0].dropna(subset=['prob_generated'])
        if not df_false_prob.empty:
            label = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[1].scatter(df_false_prob['step'], df_false_prob['prob_generated'],
                                   marker='X', color=color, s=marker_size, label=label,
                                   zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            # Add handle to dict only if label is new
            legend_handles_labels_ax1.setdefault(label, handle)
    vline_label_added = False # Flag to add vline label only once
    if poison_step_indices:
        # Ensure indices are numeric and sorted
        valid_indices = sorted([idx for idx in poison_step_indices if isinstance(idx, (int, float))])
        label_vline = 'Poison Step' # Consolidated label for legend
        for ps in valid_indices:
            # Determine label for the first plot (ax[0])
            current_label = label_vline if not vline_label_added else "_nolegend_"
            # Plot vline on ax[0] and get handle for legend
            handle = ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, label=current_label, zorder=1)
            # If this is the first vline, add its handle to the legend dict
            if not vline_label_added:
                legend_handles_labels_ax0[label_vline] = handle
                vline_label_added = True
            # Plot vline on ax[1] without adding to legend
            ax[1].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, zorder=1)

    # Generate consolidated legends using the collected handles and labels
    # Check if the dictionary is not empty before creating legend
    if legend_handles_labels_ax0:
      ax[0].legend(legend_handles_labels_ax0.values(), legend_handles_labels_ax0.keys(), loc='best', fontsize='small')
    if legend_handles_labels_ax1:
      ax[1].legend(legend_handles_labels_ax1.values(), legend_handles_labels_ax1.keys(), loc='best', fontsize='small')

    # Apply grid to main axes
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Final Adjustments ---
    # Hide X labels and ticks for the top plot (ax[0]) as it shares with ax[1]
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].tick_params(axis='x', bottom=False)

    # Adjust x-axis ticks and limits for the bottom plot (ax[1])
    if max_step_overall >= 0: # Check >= 0 to handle single step case
        # Calculate tick step aiming for ~10 ticks
        tick_step = max(1, int(np.ceil((max_step_overall + 1) / 10)))
        # Generate ticks, ensuring 0 and max_step_overall are included if relevant
        ticks = np.arange(0, max_step_overall + tick_step, tick_step)
        ticks = ticks[ticks <= max_step_overall] # Remove ticks beyond max step
        if 0 not in ticks: ticks = np.insert(ticks, 0, 0) # Ensure 0 is included
        if max_step_overall not in ticks and max_step_overall > 0: ticks = np.append(ticks, max_step_overall) # Ensure max step included

        ax[1].set_xticks(np.unique(ticks)) # Set unique ticks on the bottom axis
        ax[1].set_xlim(left=-0.5, right=max_step_overall + 0.5) # Add padding to limits
    else: # Handle case with no steps (should be rare if processed_runs isn't empty)
         ax[1].set_xticks([0])
         ax[1].set_xlim(left=-0.5, right=0.5)

    # Apply tight layout, adjusting rectangle to make space for suptitle
    try:
      fig.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust top value (0.94) if suptitle overlaps
    except ValueError:
        # tight_layout can sometimes fail with complex layouts or specific backends
        print("Warning: tight_layout failed, plot may have overlapping elements.", file=sys.stderr)

    # Optional: Confirmation print (can be removed if too verbose)
    # print(f"Plot generated successfully using {len(processed_runs)} runs.")

    return fig


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate belief analysis plot from experiment results JSON file.")
    parser.add_argument("json_file", help="Path to the experiment results JSON file.")
    parser.add_argument("-o", "--output", help="Optional: Output PNG file path. Defaults to saving in the same directory as the input JSON.", default=None)
    args = parser.parse_args()

    input_path = Path(args.json_file)
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load the JSON data
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract necessary data for plotting
    all_runs = data.get("runs", [])
    metadata = data.get("metadata", {})

    if not all_runs:
        print("Error: No 'runs' found in the JSON data.", file=sys.stderr)
        sys.exit(1)

    # Safely get query strings from the first run (assuming consistency)
    main_query = all_runs[0].get("belief_query", "N/A")
    toxic_query = all_runs[0].get("toxic_belief_query", "N/A")

    # Find poison step indices from the first poisoned run
    poison_indices = None
    for run in all_runs:
        if run.get("run_type") == "poisoned" and run.get("poison_step_indices") is not None:
            poison_indices = run.get("poison_step_indices")
            break # Use the first one found

    print(f"Generating plot for {len(all_runs)} runs from {input_path.name}...")
    # Generate the plot
    figure = plot_belief_analysis(
        all_run_results=all_runs,
        main_belief_query=main_query,
        toxic_belief_query=toxic_query,
        poison_step_indices=poison_indices,
        use_seaborn=True # Or set based on preference/availability
    )

    # Save the plot
    if figure:
        if args.output:
            output_path = Path(args.output)
            # Ensure output directory exists if specified path includes directories
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: Save in the same directory as the input file
            output_dir = input_path.parent
            output_filename = input_path.stem + "_belief_analysis.png" # e.g., experiment_results_all_runs_belief_analysis.png
            output_path = output_dir / output_filename

        try:
            figure.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(figure) # Close the figure
            print(f"Plot successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}", file=sys.stderr)
            plt.close(figure) # Still close figure on error
            sys.exit(1)
    else:
        print("Plot generation failed (returned None).", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()