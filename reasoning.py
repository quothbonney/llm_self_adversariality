# Cell 4: Modified Reasoning Chain Function - TQDM Friendly
import time # Make sure time is imported if used for delays later etc.
from llm_handling import (
    get_llm_response, get_belief_logits, generate_poison_step,
    get_logprobs_for_text, extract_target_logprobs, features_from_logprobs, enc as global_encoder # Import new helpers and encoder
)
from typing import List, Dict, Optional, Tuple, Any # Added Typing

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

# Import tqdm and tqdm.write for clean progress output
try:
    # Prefer notebook version if available
    from tqdm.notebook import tqdm as tqdm_base
    from tqdm.notebook import tqdm as tqdm_write_proxy # In notebook, write is implicitly handled
except ImportError:
    from tqdm import tqdm as tqdm_base
    from tqdm import tqdm as tqdm_write_proxy # Standard tqdm can use tqdm.write

# Helper to get the correct write function
tqdm_write = getattr(tqdm_write_proxy, 'write', print) # Use tqdm.write if available, else fallback to print

def run_reasoning_chain_with_belief_probing(
    client,
    system_prompt,
    user_task,
    max_steps,
    belief_query, # Original belief query (kept for signature compatibility)
    toxic_belief_query,
    model_id=DEFAULT_MODEL_ID,
    temperature=0.3,
    max_tokens_step=150,
    stop_sequences_step=None,
    poison_step_indices=None,
    misaligned_ai_persona=None,
    fallback_poison_string="Fallback: Intentional deviation point.",
    # --- New Parameter for Metrics ---
    metrics_capture_mode: str = 'NONE', # 'NONE', 'POISON_ONLY', 'ALL'
    # --- End New Parameter ---
    verbose=True,
    verbose_poison_gen=False,
    verbose_metrics_calc: bool = False # Verbosity for metric calculation steps
) -> Tuple[List[Dict[str, str]], Dict[int, Dict[str, Any]]]: # Modified return type hint
    """
    Runs step-by-step reasoning, probes belief state, dynamically generates
    adversarial content, and optionally calculates generation metrics.
    Optimized for cleaner tqdm output.

    Args:
        client: Initialized API client.
        system_prompt: System prompt for the reasoning AI.
        user_task: Initial user request/topic.
        max_steps: Maximum reasoning steps.
        belief_query: Original belief query (unused, kept for signature).
        toxic_belief_query: Query used for belief probing.
        model_id: LLM identifier.
        temperature: Sampling temperature.
        max_tokens_step: Max tokens per step.
        stop_sequences_step: Stop sequences.
        poison_step_indices: Steps (1-based) to poison.
        misaligned_ai_persona: Prompt for poison generation.
        fallback_poison_string: Fallback if poison generation fails.
        metrics_capture_mode: 'NONE', 'POISON_ONLY', or 'ALL'. Controls generation metric calculation.
        verbose: General verbosity.
        verbose_poison_gen: Verbosity for poison generation.
        verbose_metrics_calc: Verbosity for metric calculation.

    Returns:
        Tuple containing:
        - messages: Conversation history.
        - step_data: Dictionary mapping step number (int) to data:
            - 'belief_logits': Result from get_belief_logits.
            - 'is_poisoned': Boolean.
            - 'generation_metrics': Dict of calculated metrics (ppl, avg_lp, etc.)
                                      if calculated, otherwise None.
    """
    if not client:
        # Use tqdm_write here too, in case tqdm is active from an outer loop
        tqdm_write("ERROR: Invalid client object provided.")
        return [], {}
    if not global_encoder:
        tqdm_write("ERROR: Tiktoken encoder not available.")
        return [], {}
    if metrics_capture_mode not in ['NONE', 'POISON_ONLY', 'ALL']:
        tqdm_write(f"Warning: Invalid metrics_capture_mode '{metrics_capture_mode}'. Defaulting to 'NONE'.")
        metrics_capture_mode = 'NONE'

    if stop_sequences_step is None:
        stop_sequences_step = ["\nStep", "\nFinal Answer:", "<|eot_id|>"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task}
    ]
    # --- Modified Data Structure ---
    step_data = {} # Stores belief logits AND generation metrics per step
    # --- End Modification ---
    poison_indices_set = set(poison_step_indices) if poison_step_indices else set()
    last_action = "Init"

    if verbose:
        tqdm_write(f"\nStarting Reasoning Chain (Model: {model_id}, Temp: {temperature})")
        if poison_indices_set:
             tqdm_write(f"-> Dynamic Contextual Poisoning ENABLED: Steps {sorted(list(poison_indices_set))}")
        else:
             tqdm_write("-> Poisoning: None")
        tqdm_write(f"-> Generation Metrics Capture Mode: {metrics_capture_mode}") # Log metrics mode

    step_iterator = range(1, max_steps + 1)
    step_range = tqdm_base(step_iterator, desc="Reasoning Steps", leave=True) if verbose else step_iterator

    actual_completed_steps = 0

    for step in step_range:
        current_step_content = None
        is_poisoned_this_step = False
        generation_metrics_calculated = None # Store metrics for this step
        step_status = "OK"

        if verbose and hasattr(step_range, 'set_description'):
             step_range.set_description(f"Step {step}/{max_steps}")

        # Store context *before* this step's generation for potential metric calculation
        context_before_step = messages[:]

        # --- Check for Poisoning ---
        if step in poison_indices_set:
            last_action = f"Poison Gen (Step {step})"
            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status="Attempting")

            is_poisoned_this_step = True
            generated_poison = generate_poison_step(
                client=client,
                history_upto_n_minus_1=context_before_step, # History BEFORE this step
                topic=user_task,
                step_n=step,
                misaligned_persona_prompt=misaligned_ai_persona,
                model_id=model_id,
                verbose=verbose_poison_gen
            )

            if generated_poison:
                current_step_content = generated_poison
                step_status = "Poison OK"
            else: # Poison generation failed
                current_step_content = fallback_poison_string
                step_status = "Poison Fallback"
                if verbose: tqdm_write(f"   -> ⚠️ Failed poison gen for Step {step}. Using fallback.")

            # Add the (poisoned or fallback) content to history
            full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
            messages.append(full_step_message)

        else:
            # --- Regular Step Generation ---
            last_action = f"Step Gen (Step {step})"
            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status="Generating")

            assistant_prompt_for_step = f"Step {step}:"
            # Send history *including* the "Step N:" prompt stub
            current_messages_for_step = context_before_step + [{"role": "assistant", "content": assistant_prompt_for_step}]

            response = get_llm_response(
                client=client,
                messages=current_messages_for_step,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens_step,
                stop_sequences=stop_sequences_step,
                # No logprobs/echo needed for standard generation here
            )

            if response and response.choices and response.choices[0].message:
                step_content_raw = response.choices[0].message.content.strip()

                # Check for refusal or empty response
                if not step_content_raw or \
                   "didn't provide any input" in step_content_raw or \
                   (step_content_raw.lower().startswith("i apologize") and "next step" in step_content_raw.lower()):
                    step_status = "Refusal/Empty"
                    if verbose:
                        tqdm_write(f"\n⚠️ Step {step}: Model generated invalid/refusal content: '{step_content_raw[:50]}...'")
                        tqdm_write("   Stopping reasoning chain.")
                    break # Stop the loop

                current_step_content = step_content_raw
                step_status = "Step OK"
                # Add the *actual* generated content
                full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
                messages.append(full_step_message)

                if "final answer" in step_content_raw.lower():
                    step_status = "Premature Final Ans"
                    if verbose: tqdm_write(f"\n⚠️ WARNING: Model attempted premature final answer at step {step}. Stopping after belief probe.")
                    # Probe belief then break outer loop below
            else:
                step_status = "API Fail/No Choice"
                if verbose: tqdm_write(f"\n⚠️ WARNING: Received no content/choices for step {step}. Stopping early.")
                break # Stop if API call fails

        # --- Generation Metrics Calculation (Common to both poisoned and regular steps if enabled) ---
        should_calculate_metrics = (
            metrics_capture_mode == 'ALL' or
            (metrics_capture_mode == 'POISON_ONLY' and is_poisoned_this_step)
        )

        if should_calculate_metrics and current_step_content is not None:
            last_action = f"Metrics Calc (Step {step})"
            metrics_status = "Calculating"
            if verbose and hasattr(step_range, 'set_postfix'):
                step_range.set_postfix(action=last_action, status=metrics_status, step_outcome=step_status)
            if verbose_metrics_calc:
                 tqdm_write(f"   Calculating metrics for generated content (Step {step})...")

            # 1. Prepare context string from history *before* the current step
            context_str = "\n".join([f"{m['role']}: {m['content']}" for m in context_before_step])
            text_to_score = current_step_content # The actual generated content for this step

            # 2. Get logprobs via API
            logprobs_response = get_logprobs_for_text(
                client=client,
                model_id=model_id,
                context=context_str,
                text_to_score=text_to_score
            )
            metrics_status = "API Done"

            if logprobs_response:
                metrics_status = "Extracting LPs"
                # 3. Extract target logprobs
                target_lps = extract_target_logprobs(
                    api_response=logprobs_response,
                    context=context_str,
                    text_to_score=text_to_score,
                    encoder=global_encoder # Use the globally loaded encoder
                )

                if target_lps is not None: # Allow empty list from extract_target_logprobs
                    metrics_status = "Calculating Feats"
                    # 4. Calculate features
                    generation_metrics_calculated = features_from_logprobs(target_lps)
                    if generation_metrics_calculated:
                         metrics_status = "Metrics OK"
                         if verbose_metrics_calc:
                             ppl = generation_metrics_calculated.get('ppl', float('nan'))
                             avg_lp = generation_metrics_calculated.get('avg_lp', float('nan'))
                             var_surp = generation_metrics_calculated.get('var_surp', float('nan'))
                             tqdm_write(f"     -> Metrics OK: PPL={ppl:.2f}, "
                                        f"AvgLP={avg_lp:.4f}, "
                                        f"VarSurp={var_surp:.3f}")
                    else:
                        metrics_status = "Metrics Fail (Features)"
                        if verbose_metrics_calc: tqdm_write("     -> Metrics Calculation Failed (features_from_logprobs returned None).")
                else:
                    metrics_status = "Metrics Fail (Extract)"
                    if verbose_metrics_calc: tqdm_write("     -> Metrics Calculation Failed (extract_target_logprobs returned None).")
            else:
                metrics_status = "Metrics Fail (API)"
                if verbose_metrics_calc: tqdm_write("     -> Metrics Calculation Failed (get_logprobs_for_text returned None).")

            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status=metrics_status, step_outcome=step_status)
        # --- END METRICS CALCULATION ---

        # --- Belief Probing (Common to both poisoned and regular steps) ---
        if current_step_content is not None:
            last_action = f"Belief Probe (Post Step {step})"
            probe_status="Probing"
            if verbose and hasattr(step_range, 'set_postfix'):
                 # Show metrics status if calculated, otherwise just step outcome
                 postfix_dict = {"action": last_action, "status": probe_status, "step_outcome": step_status}
                 if 'metrics_status' in locals() and metrics_status: # Add metrics status if available
                     postfix_dict['metrics'] = metrics_status
                 step_range.set_postfix(**postfix_dict)

            logits_result = get_belief_logits(
                client=client,
                current_history=messages, # Use history *including* the current step N
                belief_query=toxic_belief_query,
                model_id=model_id,
                verbose=False # Keep belief probe itself quiet
            )
            probe_status = "Probe OK"

            # --- Store Data for this Step ---
            step_data[step] = {
                "belief_logits": logits_result,
                "is_poisoned": is_poisoned_this_step,
                "generation_metrics": generation_metrics_calculated # Will be None if not calculated or failed
            }
            # --- End Storing Data ---

            actual_completed_steps = step # Mark step as successfully processed
            final_step_status = step_status # Record status before potential break
        else:
             # Should only happen if generation failed catastrophically before content assignment
             probe_status = "Skipped (No Content)"
             final_step_status = step_status # Record the failure status
             if not is_poisoned_this_step: # Don't stop if only poison fallback occurred
                 if verbose: tqdm_write(f"\nStopping loop: No content generated for step {step} before probe.")
                 break

        # Update postfix one last time for the step
        if verbose and hasattr(step_range, 'set_postfix'):
             postfix_dict = {"action": last_action, "status": probe_status, "step_outcome": final_step_status}
             if 'metrics_status' in locals() and metrics_status: # Add metrics status if available
                  postfix_dict['metrics'] = metrics_status
             step_range.set_postfix(**postfix_dict)

        # Break if premature answer was detected after probing
        if final_step_status == "Premature Final Ans":
             break

    # --- End of Loop ---
    final_message_count = len(messages)
    if verbose:
        # Use the last recorded status
        final_loop_status = final_step_status if 'final_step_status' in locals() else step_status
        tqdm_write(f"\nReasoning chain finished or stopped after {actual_completed_steps} actual steps completed ({final_message_count} total messages).")
        if actual_completed_steps < max_steps and final_loop_status not in ["Premature Final Ans", "Step OK", "Poison OK", "Poison Fallback", "Metrics OK"]:
            tqdm_write(f"NOTE: Chain stopped early due to status '{final_loop_status}'. Max steps requested: {max_steps}.")

    # Return the history and the collected step data (belief + metrics)
    return messages, step_data # Return the modified step_data structure