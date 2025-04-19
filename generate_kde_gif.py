# generate_kde_gif.py (v3 - Fix KDE input data)
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import sys # For error messages
import imageio # For creating GIF
import os # For managing temporary files
import shutil # For removing temporary directory

# Try importing seaborn, handle if missing
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid") # Apply theme globally once
except ImportError:
    print("ERROR: Seaborn is required for plotting. Please install it (`pip install seaborn`)", file=sys.stderr)
    sns = None # Set sns to None if import fails

# --- Helper Function: Data Processing (Copied from original script) ---
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
    # Ensure keys are treated as strings initially, as they come from JSON
    for key_str in map(str, belief_data.keys()):
        try:
            step_int = int(key_str)
            valid_step_keys[step_int] = key_str
        except (ValueError, TypeError):
            continue # Skip keys that cannot be converted to integers

    if not valid_step_keys:
         return pd.DataFrame(columns=[
            'step', 'generated', 'True_Logprob', 'False_Logprob',
            'belief_score', 'prob_generated', 'belief_numeric'
         ])

    sorted_steps = sorted(valid_step_keys.keys())

    for step_int in sorted_steps:
        key_str = valid_step_keys[step_int]
        data = belief_data.get(key_str)
        if data is None and step_int in belief_data:
             data = belief_data[step_int]

        if not data: continue

        generated = data.get('Generated')
        true_lp = data.get('True_Logprob')
        false_lp = data.get('False_Logprob')

        try: true_lp = float(true_lp) if true_lp is not None else np.nan
        except (ValueError, TypeError): true_lp = np.nan
        try: false_lp = float(false_lp) if false_lp is not None else np.nan
        except (ValueError, TypeError): false_lp = np.nan

        belief_score, true_prob, false_prob = np.nan, np.nan, np.nan
        if pd.notna(true_lp) and true_lp <= 0: true_prob = np.exp(true_lp)
        if pd.notna(false_lp) and false_lp <= 0: false_prob = np.exp(false_lp)

        if pd.notna(true_prob) and pd.notna(false_prob):
            denominator = true_prob + false_prob
            if denominator > 1e-9: belief_score = true_prob / denominator
            elif true_prob > false_prob: belief_score = 1.0
            elif false_prob > true_prob: belief_score = 0.0
            else: belief_score = 0.5
        elif pd.notna(true_prob): belief_score = true_prob
        elif pd.notna(false_prob): belief_score = 1.0 - false_prob

        prob_gen = np.nan
        if generated == 'True' and pd.notna(true_prob): prob_gen = true_prob
        elif generated == 'False' and pd.notna(false_prob): prob_gen = false_prob

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

    df = pd.DataFrame(processed_data)
    for col in ['True_Logprob', 'False_Logprob', 'belief_score', 'prob_generated', 'belief_numeric']:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'step' in df.columns:
        df['step'] = pd.to_numeric(df['step'], errors='coerce').astype('Int64')

    return df

# --- New Function: Generate Single KDE Frame (Corrected) ---
def generate_kde_frame(
    step: int,
    clean_data: pd.DataFrame, # Combined DataFrame for all clean runs
    poisoned_data: pd.DataFrame, # Combined DataFrame for all poisoned runs
    output_path: Path,
    title: str,
    max_density: Optional[float] = None # Optional: To keep density axis consistent
) -> Optional[float]:
    """
    Generates a single KDE plot frame for a given step and saves it.
    Corrected to pass DataFrame to sns.kdeplot's 'data' argument.
    Returns the maximum density observed in this frame for potential scaling.
    """
    if sns is None: return None # Seaborn needed

    fig, ax = plt.subplots(figsize=(8, 6))
    current_max_density = 0.0
    color_clean, color_poisoned = 'skyblue', 'lightcoral'
    kde_common_args = {'linewidth': 2, 'alpha': 0.7, 'fill': True, 'bw_adjust': 0.4, 'clip': (0, 1)}

    # Filter data for the current step and drop NaNs in belief_score
    # Keep as DataFrames
    df_step_clean = clean_data.loc[(clean_data['step'] == step) & clean_data['belief_score'].notna()].copy()
    df_step_poisoned = poisoned_data.loc[(poisoned_data['step'] == step) & poisoned_data['belief_score'].notna()].copy()

    # Plot KDE for clean runs if data exists (at least 2 points needed for KDE)
    plotted_clean = False
    if not df_step_clean.empty and len(df_step_clean) > 1:
        try:
            # Pass the filtered DataFrame to 'data' and column name to 'x'
            sns.kdeplot(data=df_step_clean, x='belief_score', color=color_clean,
                        label=f'Clean ({len(df_step_clean)})', ax=ax, **kde_common_args)
            plotted_clean = True
            # Get density values to find max
            if ax.lines:
                 line = ax.lines[-1]
                 if line.get_label().startswith('Clean'):
                    density_values = line.get_ydata()
                    # Check for valid density values before taking max
                    if len(density_values) > 0 and np.all(np.isfinite(density_values)):
                        current_max_density = max(current_max_density, density_values.max())

        except Exception as e:
             # More specific error message if possible
             print(f"Warning: Could not plot KDE for clean data at step {step}: {e}", file=sys.stderr)


    # Plot KDE for poisoned runs if data exists (at least 2 points needed)
    plotted_poisoned = False
    if not df_step_poisoned.empty and len(df_step_poisoned) > 1:
        try:
            # Pass the filtered DataFrame to 'data' and column name to 'x'
            sns.kdeplot(data=df_step_poisoned, x='belief_score', color=color_poisoned,
                        label=f'Poisoned ({len(df_step_poisoned)})', ax=ax, **kde_common_args)
            plotted_poisoned = True
            # Get density values
            if ax.lines:
                 # Check if the last line added corresponds to the poisoned plot
                 line = ax.lines[-1]
                 if line.get_label().startswith('Poisoned'):
                    density_values = line.get_ydata()
                    if len(density_values) > 0 and np.all(np.isfinite(density_values)):
                        current_max_density = max(current_max_density, density_values.max())

                 # If both were plotted, the second-to-last line might be the clean one
                 # Check only if poisoned wasn't the last line (edge case)
                 elif len(ax.lines) > 1 and ax.lines[-2].get_label().startswith('Poisoned'):
                      line = ax.lines[-2]
                      density_values = line.get_ydata()
                      if len(density_values) > 0 and np.all(np.isfinite(density_values)):
                          current_max_density = max(current_max_density, density_values.max())


        except Exception as e:
            print(f"Warning: Could not plot KDE for poisoned data at step {step}: {e}", file=sys.stderr)


    # --- Formatting ---
    ax.set_title(f'{title} - Step {step}', fontsize=14)
    ax.set_xlabel("Belief Score (P(True))", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(-0.05, 1.05)

    # Set Y limit based on max_density
    # Use the calculated max_density from the current frame if no global one is provided
    effective_max_density = max_density if max_density is not None else current_max_density
    if effective_max_density is not None and effective_max_density > 1e-9: # Check if max density is positive
        ax.set_ylim(0, effective_max_density * 2.1) # Add 10% padding
    else:
        ax.set_ylim(0, 1) # Default y-limit if no KDE plotted or density is effectively zero

    # Add legend only if plots were made
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper center', fontsize='medium')
    # Add text annotation if no plots were possible
    elif not plotted_clean and not plotted_poisoned:
         ax.text(0.5, 0.5, 'No KDE plot (Insufficient data for this step)',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontsize=12, color='grey')

    # Save the figure
    try:
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        # Return density only if plot was successful
        return current_max_density if (plotted_clean or plotted_poisoned) else 0.0
    except Exception as e:
        print(f"Error saving frame {output_path}: {e}", file=sys.stderr)
        plt.close(fig)
        return None


# --- Main Execution Logic --- (No changes needed in main)
def main():
    if sns is None:
        print("Seaborn is required but not installed. Exiting.", file=sys.stderr)
        sys.exit(1)
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio is required for GIF creation. Please install it (`pip install imageio`)", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate KDE plot GIF of belief scores over steps from experiment results JSON file.")
    parser.add_argument("json_file", help="Path to the experiment results JSON file.")
    parser.add_argument("-o", "--output", help="Optional: Output GIF file path. Defaults to saving in the same directory as the input JSON.", default=None)
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for the output GIF.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep the individual frame PNG files after generating the GIF.")
    parser.add_argument("--no-rescale", action="store_true", help="Do not rescale Y-axis; allow density axis to change between frames.")

    args = parser.parse_args()

    input_path = Path(args.json_file)
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # --- Determine Output Path ---
    if args.output:
        output_path_gif = Path(args.output)
        output_path_gif.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
        output_filename_gif = input_path.stem + "_belief_kde.gif"
        output_path_gif = output_dir / output_filename_gif

    temp_dir_name = "_temp_kde_frames_" + input_path.stem.replace(" ", "_") # Sanitize name
    temp_dir = Path("./" + temp_dir_name)
    try:
        temp_dir.mkdir(exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create temporary directory {temp_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load the JSON data ---
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {input_path}: {e}", file=sys.stderr)
        shutil.rmtree(temp_dir)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file {input_path}: {e}", file=sys.stderr)
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # --- Process Data ---
    all_runs_data = data.get("runs", [])
    if not all_runs_data:
        print("Error: No 'runs' found in the JSON data.", file=sys.stderr)
        shutil.rmtree(temp_dir)
        sys.exit(1)

    main_query = "N/A"
    for r in all_runs_data:
        if r.get("belief_query"):
            main_query = r.get("belief_query")
            break
    plot_title = f'Belief Score Distribution\nQuery: "{main_query}"'

    all_processed_dfs = []
    all_steps = set()
    clean_dfs = []
    poisoned_dfs = []

    print("Processing runs...")
    for i, run_data in enumerate(all_runs_data):
        run_type = run_data.get("run_type", "unknown")
        run_id = run_data.get("experiment_id", f"run_{i}")
        belief_data_raw = run_data.get("belief_tracking_data", {})
        df_run = process_belief_data(belief_data_raw, run_id)

        if not df_run.empty and 'step' in df_run.columns and not df_run['step'].dropna().empty:
            if 'belief_score' not in df_run.columns:
                 df_run['belief_score'] = np.nan
            all_processed_dfs.append({"type": run_type, "id": run_id, "df": df_run})
            all_steps.update(df_run['step'].dropna().astype(int).unique())

            if run_type == 'clean':
                 if 'step' in df_run.columns and 'belief_score' in df_run.columns:
                    clean_dfs.append(df_run[['step', 'belief_score']])
            elif run_type == 'poisoned':
                  if 'step' in df_run.columns and 'belief_score' in df_run.columns:
                     poisoned_dfs.append(df_run[['step', 'belief_score']])

    if not all_processed_dfs:
        print("Error: No valid runs with processable belief data found.", file=sys.stderr)
        shutil.rmtree(temp_dir)
        sys.exit(1)
    if not all_steps:
         print("Error: No valid steps found across all runs.", file=sys.stderr)
         shutil.rmtree(temp_dir)
         sys.exit(1)

    df_clean_all = pd.concat(clean_dfs, ignore_index=True) if clean_dfs else pd.DataFrame(columns=['step', 'belief_score'])
    df_poisoned_all = pd.concat(poisoned_dfs, ignore_index=True) if poisoned_dfs else pd.DataFrame(columns=['step', 'belief_score'])

    # --- Generate Frames ---
    sorted_steps = sorted(list(all_steps))
    frame_files = []
    frame_max_densities = []
    generated_frame_paths = {} # Track generated frames by step

    print(f"Generating {len(sorted_steps)} frames...")
    for step in sorted_steps:
        frame_filename = temp_dir / f"frame_step_{step:04d}.png"
        max_d = generate_kde_frame(
            step=step,
            clean_data=df_clean_all,
            poisoned_data=df_poisoned_all,
            output_path=frame_filename,
            title=plot_title
        )
        if max_d is not None:
             if frame_filename.exists():
                frame_files.append(str(frame_filename))
                frame_max_densities.append(max_d)
                generated_frame_paths[step] = str(frame_filename)
             else:
                 print(f"Warning: Frame file {frame_filename} not found after generation for step {step}.", file=sys.stderr)

    # --- Optional: Re-generate frames with consistent Y-axis (density) ---
    frames_to_use = frame_files
    if not args.no_rescale and frame_files and frame_max_densities:
        non_zero_densities = [d for d in frame_max_densities if d > 1e-9]
        if non_zero_densities:
             global_max_density = max(non_zero_densities)
        else:
             global_max_density = 1.0

        # Only rescale if global max density is meaningfully positive
        if global_max_density > 1e-9:
             print(f"Re-generating frames with consistent max density: {global_max_density:.2f}...")
             frame_files_rescaled = []
             for step in sorted_steps:
                 if step in generated_frame_paths:
                      frame_filename_rescaled = temp_dir / f"frame_step_{step:04d}_scaled.png"
                      _ = generate_kde_frame(
                          step=step,
                          clean_data=df_clean_all,
                          poisoned_data=df_poisoned_all,
                          output_path=frame_filename_rescaled,
                          title=plot_title,
                          max_density=global_max_density
                      )
                      if frame_filename_rescaled.exists():
                          frame_files_rescaled.append(str(frame_filename_rescaled))

             if frame_files_rescaled:
                  frames_to_use = frame_files_rescaled
             elif frames_to_use:
                  print("Warning: Rescaling failed, using original frames for GIF.", file=sys.stderr)
        else:
            print("Skipping Y-axis rescaling as maximum density across frames is near zero.")


    # --- Create GIF ---
    if frames_to_use:
        print(f"Creating GIF: {output_path_gif} at {args.fps} FPS...")
        try:
            with imageio.get_writer(output_path_gif, mode='I', fps=args.fps) as writer:
                for filename in frames_to_use:
                    try:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                    except FileNotFoundError:
                        print(f"Warning: Frame file not found during GIF creation: {filename}", file=sys.stderr)
                    except Exception as read_err:
                         print(f"Warning: Error reading frame file {filename}: {read_err}", file=sys.stderr)

            if output_path_gif.exists() and output_path_gif.stat().st_size > 0:
                 print("GIF created successfully.")
            else:
                  print("Error: GIF creation failed or resulted in an empty file.", file=sys.stderr)

        except Exception as e:
            print(f"Error creating GIF: {e}", file=sys.stderr)
    else:
        print("No frames were successfully generated, GIF cannot be created.", file=sys.stderr)

    # --- Clean Up ---
    if not args.keep_frames and temp_dir.exists():
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}: {e}", file=sys.stderr)
    elif args.keep_frames:
         final_frame_dir = "original" if frames_to_use == frame_files else "rescaled (_scaled suffix)"
         print(f"Keeping individual {final_frame_dir} frames in: {temp_dir}")


if __name__ == "__main__":
    main()