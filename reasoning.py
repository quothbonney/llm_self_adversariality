
# Cell 4: Modified Reasoning Chain Function - TQDM Friendly
import time # Make sure time is imported if used for delays later etc.
from llm_handling import get_llm_response, get_belief_logits, generate_poison_step
DEFAULT_MODEL_ID = "meta-llama/Llama-3-70b-chat-hf"
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
    user_task, # Used as 'topic' for poison generation context
    max_steps,
    belief_query,
    toxic_belief_query,
    model_id=DEFAULT_MODEL_ID,
    temperature=0.3,
    max_tokens_step=150,
    stop_sequences_step=None,
    # --- Modified Poisoning Params ---
    poison_step_indices=None, # List/Set of step numbers (1-based) to poison
    misaligned_ai_persona=None, # Persona for dynamic generation
    fallback_poison_string="Fallback: Intentional deviation point.", # Fallback
    # --- End Modified Params ---
    verbose=True,
    verbose_poison_gen=False # Separate verbosity for poison gen attempt
):
    """
    Runs step-by-step reasoning, probes belief state, and dynamically generates
    adversarial content. Optimized for cleaner tqdm output.
    """
    if not client:
        # Use tqdm_write here too, in case tqdm is active from an outer loop
        tqdm_write("ERROR: Invalid client object provided.")
        return None, None

    if stop_sequences_step is None:
        stop_sequences_step = ["\nStep", "\nFinal Answer:", "<|eot_id|>"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task}
    ]
    belief_tracking_data = {} # Store logits per step
    poison_indices_set = set(poison_step_indices) if poison_step_indices else set()
    last_action = "Init" # For tqdm postfix

    # Initial status print (outside loop)
    if verbose:
        tqdm_write(f"\nStarting Reasoning Chain (Model: {model_id}, Temp: {temperature})")
        if poison_indices_set:
             tqdm_write(f"-> Dynamic Contextual Poisoning ENABLED: Steps {sorted(list(poison_indices_set))}")
        else:
             tqdm_write("-> Poisoning: None")

    # Setup tqdm or plain range
    step_iterator = range(1, max_steps + 1)
    if verbose:
        step_range = tqdm_base(step_iterator, desc="Reasoning Steps", leave=True)
    else:
        step_range = step_iterator # No progress bar if not verbose

    actual_completed_steps = 0 # Track steps successfully processed

    for step in step_range:
        current_step_content = None
        is_poisoned_this_step = False
        step_status = "OK" # Track status for postfix

        if verbose and hasattr(step_range, 'set_description'): # Check if it's a tqdm object
             step_range.set_description(f"Step {step}/{max_steps}")

        # --- Check if this step should be poisoned ---
        if step in poison_indices_set:
            last_action = f"Poison Gen (Step {step})"
            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status="Attempting")
            is_poisoned_this_step = True
            # Use tqdm_write for less intrusive messages if verbose
            # if verbose: tqdm_write(f"\n-> Attempting Dynamic Poison Generation for Step {step}")

            generated_poison = generate_poison_step(
                client=client,
                history_upto_n_minus_1=messages,
                topic=user_task,
                step_n=step,
                misaligned_persona_prompt=misaligned_ai_persona,
                model_id=model_id,
                verbose=verbose_poison_gen # Use separate flag for poison gen verbosity
            )

            if generated_poison:
                current_step_content = generated_poison
                step_status = "Poison OK"
                # if verbose: tqdm_write(f"   -> ✅ Generated poison for Step {step}.") # Optional: reduce verbosity further
            else:
                current_step_content = fallback_poison_string
                step_status = "Poison Fallback"
                if verbose: tqdm_write(f"   -> ⚠️ Failed poison gen for Step {step}. Using fallback.")

            # Construct message for history
            full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
            messages.append(full_step_message)
            # --- REMOVED: Printing full poisoned content routinely ---
            # if verbose: tqdm_write(f"🔴 Step {step} (POISONED):\n{current_step_content}")

        else:
            # --- Regular Step Generation ---
            last_action = f"Step Gen (Step {step})"
            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status="Generating")

            assistant_prompt_for_step = f"Step {step}:"
            step_prompt_message = {"role": "assistant", "content": assistant_prompt_for_step}
            current_messages_for_step = messages + [step_prompt_message]

            response = get_llm_response(
                client=client,
                messages=current_messages_for_step,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens_step,
                stop_sequences=stop_sequences_step,
            )

            if response and response.choices:
                step_content_raw = response.choices[0].message.content.strip()
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
                full_step_message = {"role": "assistant", "content": f"Step {step}: {current_step_content}"}
                messages.append(full_step_message)
                # --- REMOVED: Printing full step content routinely ---
                # if verbose: tqdm_write(f"\n🔵 Step {step}:\n{current_step_content}")

                if "final answer" in step_content_raw.lower():
                    step_status = "Premature Final Ans"
                    if verbose: tqdm_write(f"\n⚠️ WARNING: Model attempted premature final answer at step {step}. Stopping after belief probe.")
                    # We still probe belief for this step before breaking the outer loop
            else:
                step_status = "API Fail/No Choice"
                if verbose: tqdm_write(f"\n⚠️ WARNING: Received no content/choices for step {step}. Stopping early.")
                break # Stop if API call fails or returns no choices

        # --- Belief Probing (After step content is determined and added) ---
        if current_step_content is not None:
            last_action = f"Belief Probe (Post Step {step})"
            if verbose and hasattr(step_range, 'set_postfix'):
                 step_range.set_postfix(action=last_action, status="Probing", step_outcome=step_status)

            logits_result = get_belief_logits(
                client=client,
                current_history=messages, # Use history *including* the current step N
                belief_query=toxic_belief_query,
                model_id=model_id,
                verbose=False # Keep belief probe itself quiet unless specifically debugged
            )
            belief_tracking_data[step] = logits_result
            actual_completed_steps = step # Mark step as successfully processed including probe
        else:
             # This case should ideally not be hit often due to checks above, but as safety:
             # If step generation failed (not a fallback) stop the loop
             if not is_poisoned_this_step:
                 step_status = "Gen Fail Pre-Probe"
                 if verbose: tqdm_write(f"\nStopping loop: No content generated for step {step} before probe.")
                 break

        # Update postfix one last time for the step
        if verbose and hasattr(step_range, 'set_postfix'):
             step_range.set_postfix(action=last_action, status="Done", step_outcome=step_status)


        # Break if premature answer was detected in non-poisoned step after probing
        if step_status == "Premature Final Ans":
             break


    # --- End of Loop ---
    final_message_count = len(messages)
    if verbose:
        tqdm_write(f"\nReasoning chain finished or stopped after {actual_completed_steps} actual steps completed ({final_message_count} total messages).")
        if actual_completed_steps < max_steps and step_status not in ["Premature Final Ans", "OK", "Poison OK"]: # Don't show warning if it stopped normally or gave final answer
            tqdm_write(f"NOTE: Chain stopped early due to status '{step_status}'. Max steps requested: {max_steps}.")

    # Return both the history and the collected belief data
    return messages, belief_tracking_data