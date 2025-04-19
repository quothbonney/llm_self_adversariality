# Cell 4: Modified Reasoning Chain Function - TQDM Friendly
import time # Make sure time is imported if used for delays later etc.
import re # Import regex for confusion detection
from llm_handling import (
    get_llm_response, get_belief_logits, generate_poison_step,
    get_logprobs_for_text, extract_target_logprobs, features_from_logprobs, enc as global_encoder # Import new helpers and encoder
)
from typing import List, Dict, Optional, Tuple, Any # Added Typing
import logging # Added logging
import threading # Added threading

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
MAX_STEP_RETRIES = 3 # Maximum retries for a single step generation if confusion is detected

# Remove tqdm imports as verbose=False will be used from main.py, and we use logging now
# try:
#     # Prefer notebook version if available
#     from tqdm.notebook import tqdm as tqdm_base
#     from tqdm.notebook import tqdm as tqdm_write_proxy # In notebook, write is implicitly handled
# except ImportError:
#     from tqdm import tqdm as tqdm_base
#     from tqdm import tqdm as tqdm_write_proxy # Standard tqdm can use tqdm.write

# Helper to get the correct write function - REMOVED, using logging now
# tqdm_write = getattr(tqdm_write_proxy, 'write', print) # Use tqdm.write if available, else fallback to print

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
    max_step_retries: int = MAX_STEP_RETRIES, # <<< Add max retries parameter
    verbose=True,
    verbose_poison_gen=False,
    verbose_metrics_calc: bool = False, # Verbosity for metric calculation steps
    # --- Added progress tracking args ---
    run_progress: Optional[Dict[str, Any]] = None,
    progress_lock: Optional[threading.Lock] = None,
    experiment_id: Optional[str] = None
    # --- End added args ---
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
        run_progress: Shared dictionary to report progress.
        progress_lock: Lock for accessing run_progress.
        experiment_id: Unique ID for the current run.

    Returns:
        Tuple containing:
        - messages: Conversation history.
        - step_data: Dictionary mapping step number (int) to data:
            - 'belief_logits': Result from get_belief_logits.
            - 'is_poisoned': Boolean.
            - 'generation_metrics': Dict of calculated metrics (ppl, avg_lp, etc.)
                                      if calculated, otherwise None.
            - 'status': String indicating the final status of the step ('OK', 'Poison OK', 'Failed Retries', etc.)
    """
    if not client:
        # Use logging instead of tqdm_write
        logging.error("ERROR: Invalid client object provided.")
        return [], {}
    if not global_encoder:
        logging.error("ERROR: Tiktoken encoder not available.")
        return [], {}
    if metrics_capture_mode not in ['NONE', 'POISON_ONLY', 'ALL']:
        logging.warning(f"Warning: Invalid metrics_capture_mode '{metrics_capture_mode}'. Defaulting to 'NONE'.")
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

    # --- Check if progress tracking is enabled ---
    progress_tracking_enabled = run_progress is not None and progress_lock is not None and experiment_id is not None
    # --- End check ---

    if verbose:
        logging.info(f"\nStarting Reasoning Chain (Model: {model_id}, Temp: {temperature})")
        if poison_indices_set:
             logging.info(f"-> Dynamic Contextual Poisoning ENABLED: Steps {sorted(list(poison_indices_set))}")
        else:
             logging.info("-> Poisoning: None")
        logging.info(f"-> Generation Metrics Capture Mode: {metrics_capture_mode}")

    step_iterator = range(1, max_steps + 1)

    actual_completed_steps = 0
    chain_stopped_early_reason = None # Store reason if chain stops prematurely

    for step in step_iterator:
        current_step_content = None
        is_poisoned_this_step = False
        generation_metrics_calculated = None # Store metrics for this step
        step_status = "Pending" # Initial status
        final_step_status = "Pending" # Status before potential break

        # Store context *before* this step's generation for potential metric calculation
        context_before_step = messages[:]

        # --- Check for Poisoning ---
        if step in poison_indices_set:
            last_action = f"Poison Gen (Step {step})"
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
                if verbose: logging.warning(f"   -> ⚠️ Failed poison gen for Step {step}. Using fallback.")

            # Add the (poisoned or fallback) content to history
            full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
            messages.append(full_step_message)

        else:
            # --- Regular Step Generation (with Retry Logic) ---
            last_action = f"Step Gen (Step {step})"
            retry_count = 0
            step_content_raw = None
            generation_successful = False

            while retry_count <= max_step_retries:
                assistant_prompt_for_step = f"Step {step}:"
                # Send history *including* the "Step N:" prompt stub
                current_messages_for_step = context_before_step + [{"role": "assistant", "content": assistant_prompt_for_step}]

                if retry_count > 0:
                     last_action = f"Step Gen Retry {retry_count} (Step {step})"
                     # Optional: Add a clarifying message on retry?
                     # current_messages_for_step.insert(-1, {"role": "user", "content": f"Please generate the content for Step {step} based on the history. Focus on the current step."})
                     logging.warning(f"   -> Retrying Step {step} generation (Attempt {retry_count}/{max_step_retries})...")


                response = get_llm_response(
                    client=client,
                    messages=current_messages_for_step,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens_step,
                    stop_sequences=stop_sequences_step,
                )

                if response and response.choices and response.choices[0].message:
                    step_content_raw = response.choices[0].message.content.strip()

                    # --- Confusion Detection Logic ---
                    is_confused = False
                    confused_reason = ""

                    # 1. Check for empty/generic refusal
                    if not step_content_raw or "didn't provide any input" in step_content_raw:
                        is_confused = True
                        confused_reason = "Empty or generic refusal content"

                    # 2. Check for explicit prompt requests or apologies about *previous* steps
                    # Example patterns: "Please provide...Step N", "I apologize...Step N-1"
                    prompt_request_pattern = r"(please|could you)?\s*(provide|give me)\s*(the)?\s*(prompt|topic|input)\s*(for|to continue with)?\s*step\s*\d+"
                    apology_pattern = r"i apologize.*(didn't|did not)\s*(receive|get)\s*(a|the)?\s*(prompt|topic|input)\s*(for|to continue with)?\s*step\s*\d+" # Matches apologies about not receiving prompts
                    mention_wrong_step_pattern = rf"step\s*{step-1}" # Simple check for mentioning previous step explicitly

                    if not is_confused and (re.search(prompt_request_pattern, step_content_raw, re.IGNORECASE) or re.search(apology_pattern, step_content_raw, re.IGNORECASE)):
                        is_confused = True
                        confused_reason = "Explicitly asked for prompt or apologized for missing previous input"
                        # More specific check: Did it mention the *current* step number in the request/apology?
                        current_step_mention = rf"(step\s*{step}|step\s*{step-1})" # Check if it mentions current or previous step in request/apology
                        if re.search(current_step_mention, step_content_raw, re.IGNORECASE):
                             confused_reason += f" (mentioned step {step} or {step-1})"


                    # 3. Basic repetition check (optional, can be noisy)
                    # if not is_confused and step > 1 and messages[-1]['role'] == 'assistant':
                    #     prev_step_content = messages[-1]['content'].split(f"Step {step-1}:", 1)[-1].strip()
                    #     if len(step_content_raw) > 20 and step_content_raw.lower() == prev_step_content.lower():
                    #         is_confused = True
                    #         confused_reason = "Repeated previous step content"

                    # --- End Confusion Detection ---

                    if is_confused:
                        logging.warning(f"   -> ⚠️ Step {step} Confusion Detected (Attempt {retry_count}): {confused_reason}. Content: '{step_content_raw[:100]}...'")
                        retry_count += 1
                        step_status = f"Retry {retry_count-1} Failed"
                        time.sleep(0.5) # Small delay before retry
                        continue # Go to next iteration of the while loop

                    # If not confused, generation is successful for this attempt
                    current_step_content = step_content_raw
                    step_status = "Step OK"
                    generation_successful = True
                    break # Exit the retry loop

                else: # API call failed or returned no choices
                    logging.warning(f"   -> ⚠️ API call failed or no choices returned for Step {step} (Attempt {retry_count}).")
                    retry_count += 1
                    step_status = f"Retry {retry_count-1} Failed (API Error)"
                    time.sleep(1.0) # Longer delay for API errors
                    continue # Go to next iteration of the while loop

            # --- After Retry Loop ---
            if not generation_successful:
                step_status = "Failed Retries"
                logging.error(f"\n❌ ERROR: Step {step} failed generation after {max_step_retries} retries due to persistent confusion or API errors.")
                chain_stopped_early_reason = f"Step {step}: Failed after {max_step_retries} retries ({step_status})"
                # Add a placeholder message to history indicating failure? Optional.
                # messages.append({"role": "assistant", "content": f"Step {step}: [Generation failed after retries]"})
                break # Stop the main 'for step' loop

            # Add the successful generated content to history
            full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
            messages.append(full_step_message)

            # Check for premature final answer after successful generation
            if "final answer" in current_step_content.lower():
                step_status = "Premature Final Ans"
                if verbose: logging.warning(f"\n⚠️ WARNING: Model attempted premature final answer at step {step}. Stopping after belief probe.")
                # Probe belief then break outer loop below (handled by loop structure)


        # --- Generation Metrics Calculation (Common to both poisoned and regular steps if enabled) ---
        should_calculate_metrics = (
            metrics_capture_mode == 'ALL' or
            (metrics_capture_mode == 'POISON_ONLY' and is_poisoned_this_step)
        )

        if should_calculate_metrics and current_step_content is not None:
            last_action = f"Metrics Calc (Step {step})"
            metrics_status = "Calculating"
            if verbose_metrics_calc:
                 logging.info(f"   Calculating metrics for generated content (Step {step})...")

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
                             logging.info(f"     -> Metrics OK: PPL={ppl:.2f}, "
                                        f"AvgLP={avg_lp:.4f}, "
                                        f"VarSurp={var_surp:.3f}")
                    else:
                        metrics_status = "Metrics Fail (Features)"
                        if verbose_metrics_calc: logging.warning("     -> Metrics Calculation Failed (features_from_logprobs returned None).")
                else:
                    metrics_status = "Metrics Fail (Extract)"
                    if verbose_metrics_calc: logging.warning("     -> Metrics Calculation Failed (extract_target_logprobs returned None).")
            else:
                metrics_status = "Metrics Fail (API)"
                if verbose_metrics_calc: logging.warning("     -> Metrics Calculation Failed (get_logprobs_for_text returned None).")

        # --- Belief Probing (Common to both poisoned and regular steps) ---
        # Only probe if content was successfully generated or poison fallback was used
        if current_step_content is not None:
            last_action = f"Belief Probe (Post Step {step})"
            probe_status="Probing"
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
                "generation_metrics": generation_metrics_calculated, # Will be None if not calculated or failed
                "status": step_status # Store the final status of the step generation/poisoning
            }
            # --- End Storing Data ---

            actual_completed_steps = step # Mark step as successfully processed
            final_step_status = step_status # Record status before potential break

            # --- Update shared progress dictionary --- 
            if progress_tracking_enabled:
                try:
                    with progress_lock: # type: ignore
                         # --- Fetch existing data, update step, put back --- 
                         current_run_data = run_progress.get(experiment_id, {}) # type: ignore
                         current_run_data['step'] = step
                         # Ensure total is also present if it wasn't (should be, but safe)
                         if 'total' not in current_run_data:
                            current_run_data['total'] = max_steps
                         run_progress[experiment_id] = current_run_data # type: ignore
                         # --- End update logic ---
                except Exception as e:
                    # Log error but don't crash the reasoning chain
                    logging.error(f"[Progress Update Error] Failed for run {experiment_id}: {e}", exc_info=False)
            # --- End progress update ---

        else:
             # Should only happen if generation failed catastrophically (e.g., max retries exceeded before content assignment)
             probe_status = "Skipped (No Content)"
             final_step_status = step_status # Record the failure status
             # Log failure details if not already logged by retry loop
             if step_status == "Failed Retries" and verbose:
                 logging.warning(f"\nStopping loop: Step {step} failed after max retries before probe.")
             elif verbose:
                 logging.warning(f"\nStopping loop: No content generated or assigned for step {step} (Status: {step_status}). Cannot probe belief.")
             # No need to break here, the break happens within the retry logic if needed

        # --- Check if loop should terminate based on status ---
        if step_status == "Failed Retries":
            # Already broken inside the retry logic
            pass
        elif step_status == "Premature Final Ans":
             if verbose: logging.info(f"   -> Stopping after Step {step} due to premature final answer.")
             chain_stopped_early_reason = f"Step {step}: Premature Final Answer"
             break # Stop after probing belief for this step

    # --- End of Loop ---
    final_message_count = len(messages)
    if verbose:
        # Use the reason if stopped early, otherwise use the final status
        if chain_stopped_early_reason:
             logging.info(f"\nReasoning chain stopped early. Reason: {chain_stopped_early_reason}")
        logging.info(f"Reasoning chain finished processing up to step {actual_completed_steps} ({final_message_count} total messages). Max steps requested: {max_steps}.")

    # Return the history and the collected step data (belief + metrics + status)
    return messages, step_data # Return the modified step_data structure