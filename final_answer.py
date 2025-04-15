import re
from llm_handling import get_llm_response

DEFAULT_MODEL_ID = "meta-llama/Llama-3-70b-chat-hf"

def get_final_answer(
    client,
    messages,
    model_id=DEFAULT_MODEL_ID,
    temperature=0.3,
    max_tokens_final=300,
    stop_sequences_final=None,
    verbose=True
):
    """
    Generates the final answer based on the preceding reasoning chain.
    Uses a user request prompt. (Assumes previous refined version)
    """
    if not client or not messages:
        print("ERROR (get_final_answer): Invalid client or empty message history.")
        return None, messages

    cleaned_messages = [m for m in messages if m.get("content", "").strip()]
    if not cleaned_messages:
         print("ERROR (get_final_answer): Message history empty after cleaning.")
         return None, messages

    if stop_sequences_final is None:
        stop_sequences_final = ["<|eot_id|>", "I apologize", "I didn't receive"]

    if verbose:
        print("\nGenerating Final Answer...")

    final_answer_request_content = "Based on all the preceding reasoning steps, please now generate the Final Answer. Synthesize the key points."
    final_answer_request_message = {"role": "user", "content": final_answer_request_content}
    current_messages = cleaned_messages + [final_answer_request_message]

    response = get_llm_response(
        client=client,
        messages=current_messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens_final,
        stop_sequences=stop_sequences_final
    )

    if response and response.choices:
        final_answer_content_raw = response.choices[0].message.content.strip()
        refusal_pattern = r"^(I apologize|I didn't receive).*(Final Answer:|provide the prompt)"
        if not final_answer_content_raw or re.search(refusal_pattern, final_answer_content_raw, re.IGNORECASE | re.DOTALL):
            if verbose: print(f"\n⚠️ Final Answer: Model generated refusal/empty: '{final_answer_content_raw[:150]}...'")
            return None, messages
        else:
            final_message = {"role": "assistant", "content": f"Final Answer: {final_answer_content_raw}"}
            messages.append(final_message)
            if verbose: print(f"\n✅ Final Answer:\n{final_answer_content_raw}")
            return final_answer_content_raw, messages
    else:
        if verbose: print("WARNING (get_final_answer): Received no choices/content.")
        return None, messages

# --- Poison Generation Function (Mostly Unchanged from previous version) ---
