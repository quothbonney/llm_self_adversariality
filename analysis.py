# Necessary imports (ensure these are at the top of your analysis.py)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Needed for custom grid layout
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
# Import seaborn - it's now required for the KDE plots
try:
    import seaborn as sns
except ImportError:
    print("ERROR: Seaborn is required for the marginal distribution plots. Please install it (`pip install seaborn`)")
    sns = None # Set sns to None if import fails

# --- Helper Function: Data Processing (Assumed to be the same) ---
def process_belief_data(belief_data: Optional[Dict[int, Dict[str, Any]]], run_name: str = "Unknown") -> pd.DataFrame:
    """
    Converts a dictionary of belief data per step into a Pandas DataFrame.
    (Function code is omitted here for brevity, assumed to be the same as previously provided)
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


# --- Complete Rewritten Visualization Function ---

def plot_belief_analysis(
    all_run_results: List[Dict[str, Any]],
    main_belief_query: str,
    toxic_belief_query: str,
    poison_step_indices: Optional[List[int]] = None,
    use_seaborn: bool = True # Flag remains, but seaborn is now needed for KDE
) -> Optional[plt.Figure]:
    """
    Generates a two-panel plot with side-by-side marginal distributions
    for initial and final steps, showing belief trends across multiple runs.

    Args:
        all_run_results: List of completed ExperimentRun dictionaries.
        main_belief_query: The main query string for the overall figure title.
        toxic_belief_query: The toxic query string for the belief score panel title.
        poison_step_indices: List of step indices where poisoning was applied.
        use_seaborn: If True, attempts to use Seaborn styling for main plots.
                     Seaborn is required for marginal KDE plots regardless.

    Returns:
        matplotlib.figure.Figure: The generated plot figure object, or None if no data.
    """
    # Check if seaborn is available
    if sns is None:
        print("Seaborn import failed. Cannot generate marginal plots.")
        # Optionally, you could fall back to the non-marginal version here
        # or simply return None. Let's return None.
        return None

    # --- Plot Style Setup ---
    if use_seaborn:
        sns.set_theme(style="whitegrid")
    else:
         plt.style.use('default')

    # --- Process Data & Extract Initial/Final Step Values ---
    processed_runs = []
    # Store data for marginal plots: [value1, value2, ...]
    first_step_data = {'clean': {'belief_score': [], 'prob_generated': []},
                       'poisoned': {'belief_score': [], 'prob_generated': []}}
    final_step_data = {'clean': {'belief_score': [], 'prob_generated': []},
                       'poisoned': {'belief_score': [], 'prob_generated': []}}
    max_step_overall = 0
    num_clean = 0
    num_poisoned = 0

    if not all_run_results: return None # Handle empty input

    for i, run_data in enumerate(all_run_results):
        run_type = run_data.get("run_type", "unknown")
        run_id = run_data.get("experiment_id", f"run_{i}")
        belief_data = run_data.get("belief_tracking_data")
        df_run = process_belief_data(belief_data, run_id)

        if not df_run.empty and 'step' in df_run.columns and not df_run['step'].empty:
            processed_runs.append({"type": run_type, "id": run_id, "df": df_run})
            current_max_step = df_run['step'].max()
            current_min_step = df_run['step'].min()
            max_step_overall = max(max_step_overall, current_max_step)

            # Extract first and final step data
            first_row = df_run[df_run['step'] == current_min_step].iloc[0]
            last_row = df_run[df_run['step'] == current_max_step].iloc[0]

            if run_type in final_step_data: # Check if type is 'clean' or 'poisoned'
                # First step data
                if pd.notna(first_row['belief_score']):
                    first_step_data[run_type]['belief_score'].append(first_row['belief_score'])
                if pd.notna(first_row['prob_generated']):
                    first_step_data[run_type]['prob_generated'].append(first_row['prob_generated'])
                # Last step data
                if pd.notna(last_row['belief_score']):
                    final_step_data[run_type]['belief_score'].append(last_row['belief_score'])
                if pd.notna(last_row['prob_generated']):
                    final_step_data[run_type]['prob_generated'].append(last_row['prob_generated'])

            if run_type == 'clean': num_clean += 1
            elif run_type == 'poisoned': num_poisoned +=1

    if not processed_runs: return None # No valid runs processed

    print(f"\nProcessed {len(processed_runs)} runs ({num_clean} clean, {num_poisoned} poisoned) for plotting.")

    # --- Visualization Setup with GridSpec ---
    fig = plt.figure(figsize=(15, 9)) # Slightly taller figure
    # Grid: 2 rows, 3 columns. Marginals (cols 1, 2) are thin.
    gs = fig.add_gridspec(2, 3, width_ratios=(12, 1, 1), height_ratios=(1, 1),
                          wspace=0.05, hspace=0.2) # Adjusted spacing slightly

    ax = [None, None] # Main axes
    ax_marg_first = [None, None] # First step marginal axes
    ax_marg_last = [None, None]  # Last step marginal axes

    # Create axes using GridSpec
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[1, 0], sharex=ax[0]) # Share X axis for main plots
    ax_marg_first[0] = fig.add_subplot(gs[0, 1], sharey=ax[0]) # Share Y axis
    ax_marg_last[0] = fig.add_subplot(gs[0, 2], sharey=ax[0]) # Share Y axis
    ax_marg_first[1] = fig.add_subplot(gs[1, 1], sharey=ax[1]) # Share Y axis
    ax_marg_last[1] = fig.add_subplot(gs[1, 2], sharey=ax[1]) # Share Y axis

    # Add Overall Figure Title (Suptitle)
    fig.suptitle(f'Belief Analysis for Query: "{main_belief_query}"', fontsize=16, y=0.97)

    # --- Plotting Parameters ---
    marker_size = 60
    line_alpha = 0.6
    marker_alpha = 0.7
    color_clean = 'skyblue'
    color_poisoned = 'lightcoral'
    kde_common_args = {'linewidth': 1.5, 'alpha': 0.7, 'fill': True} # Added fill=True

    # --- Plot Marginal Distributions (Filled KDEs, No Legends) ---
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
    for i in range(2): # Loop through panels
        for ax_m in [ax_marg_first[i], ax_marg_last[i]]:
            ax_m.tick_params(axis='y', labelleft=False, left=False) # Hide y-ticks and labels
            ax_m.tick_params(axis='x', labelbottom=False, bottom=False) # Hide x-ticks and labels
            ax_m.grid(False) # Turn off grid
            ax_m.set_xlabel('') # Remove x-label ("Density")
            ax_m.set_ylabel('') # Remove y-label

    # --- Plot Time Series Data on Main Axes ---
    ax[0].set_title(f'Belief Score (P(True)) for Toxic Query: "{toxic_belief_query}"', fontsize=11) # Subtitle
    ax[0].set_ylabel("Belief Score (P(True))")
    ax[0].set_ylim(-0.05, 1.05)
    ax[1].set_title('Probability of Generated Answer (P(Generated Token))', fontsize=11) # Subtitle
    ax[1].set_xlabel("Reasoning Step")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(-0.05, 1.05)

    # Flags for consolidated legends on main plots
    legend_handles_labels_ax0 = {}
    legend_handles_labels_ax1 = {}

    for run in processed_runs:
        df = run["df"]
        run_type = run["type"]
        color = color_clean if run_type == "clean" else color_poisoned
        linestyle = '-' if run_type == "clean" else '--'

        # --- Plotting on ax[0] (Belief Score) ---
        belief_data_clean = df[['step', 'belief_score']].dropna()
        if not belief_data_clean.empty:
            label_line = f"{run_type.capitalize()} Run (Belief)"
            handle, = ax[0].plot(belief_data_clean['step'], belief_data_clean['belief_score'], marker='.', markersize=5, linestyle=linestyle,
                                 label=label_line, color=color, alpha=line_alpha, zorder=2)
            if label_line not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_line] = handle

        df_true = df[df['belief_numeric'] == 1].dropna(subset=['belief_score'])
        if not df_true.empty:
            label_marker_true = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[0].scatter(df_true['step'], df_true['belief_score'], marker='P', color=color, s=marker_size, label=label_marker_true, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_true not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_marker_true] = handle

        df_false = df[df['belief_numeric'] == 0].dropna(subset=['belief_score'])
        if not df_false.empty:
            label_marker_false = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[0].scatter(df_false['step'], df_false['belief_score'], marker='X', color=color, s=marker_size, label=label_marker_false, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_false not in legend_handles_labels_ax0: legend_handles_labels_ax0[label_marker_false] = handle

        # --- Plotting on ax[1] (Generated Prob) ---
        prob_gen_data_clean = df[['step', 'prob_generated']].dropna()
        if not prob_gen_data_clean.empty:
            label_line_ax1 = f"{run_type.capitalize()} Run (P(Gen))"
            handle, = ax[1].plot(prob_gen_data_clean['step'], prob_gen_data_clean['prob_generated'], marker='.', markersize=5, linestyle=linestyle,
                                 label=label_line_ax1, color=color, alpha=line_alpha, zorder=2)
            if label_line_ax1 not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_line_ax1] = handle

        df_true_ax1 = df[df['belief_numeric'] == 1].dropna(subset=['prob_generated'])
        if not df_true_ax1.empty:
            label_marker_true_ax1 = f'{run_type.capitalize()}: Gen="True"'
            handle = ax[1].scatter(df_true_ax1['step'], df_true_ax1['prob_generated'], marker='P', color=color, s=marker_size, label=label_marker_true_ax1, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_true_ax1 not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_marker_true_ax1] = handle

        df_false_ax1 = df[df['belief_numeric'] == 0].dropna(subset=['prob_generated'])
        if not df_false_ax1.empty:
            label_marker_false_ax1 = f'{run_type.capitalize()}: Gen="False"'
            handle = ax[1].scatter(df_false_ax1['step'], df_false_ax1['prob_generated'], marker='X', color=color, s=marker_size, label=label_marker_false_ax1, zorder=3, alpha=marker_alpha, edgecolors='black', linewidth=0.5)
            if label_marker_false_ax1 not in legend_handles_labels_ax1: legend_handles_labels_ax1[label_marker_false_ax1] = handle

    # --- Add Vertical Lines, Legends, Grid to Main Axes ---
    vline_label_added = False
    if poison_step_indices:
        sorted_indices = sorted(list(set(poison_step_indices)))
        label_vline = 'Poison Step'
        for ps in sorted_indices:
            current_label = label_vline if not vline_label_added else "_nolegend_"
            # Add to legend dict if first time
            if not vline_label_added:
                 handle = ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, label=current_label, zorder=1)
                 legend_handles_labels_ax0[label_vline] = handle
                 vline_label_added = True
            else:
                 ax[0].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, zorder=1)
            # Plot on second axis without label
            ax[1].axvline(x=ps, color='grey', linestyle=':', linewidth=1.5, zorder=1)


    # Generate consolidated legends
    ax[0].legend(legend_handles_labels_ax0.values(), legend_handles_labels_ax0.keys(), loc='best', fontsize='small')
    ax[1].legend(legend_handles_labels_ax1.values(), legend_handles_labels_ax1.keys(), loc='best', fontsize='small')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Final Adjustments ---
    plt.setp(ax[0].get_xticklabels(), visible=False) # Hide X labels for top main plot
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
         ax[1].set_xticks([0])
         ax[1].set_xlim(left=-0.5, right=0.5)

    # Adjust layout tightly, making space for suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust top of rect for suptitle

    return fig