import ollama


def budget_forcing(question, model="llama3.2:1b", max_iterations=3):
    # Phase 1: Generate initial chain-of-thought and final answer.
    response_text = (
        f"{question}\n\n"
        "Please provide a step-by-step explanation of your reasoning."
    )

    previous_output = None
    for i in range(max_iterations):
        print(f"Iteration {i + 1}: Sending to model...")

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": response_text}]
        )
        model_output = response['message']['content']
        print(f"Model Response:\n{model_output}\n")

        # If the answer hasn't changed from the previous iteration, assume convergence.
        if previous_output and previous_output.strip() == model_output.strip():
            print("Answer converged; stopping early.\n")
            break

        previous_output = model_output

        # Append the explicit "Wait" prompt to force further evaluation.
        response_text = (
            f"Wait. Let's read the question carefully. The question asked: \"{question}\". "
            f"You replied: \"{model_output}\". Break down the process of answering the question."
        )

    print("Max iterations reached, returning last response.")
    return model_output  # Expected to include the final answer.


# Example tests:
test_questions = [
    "How many r's are in the word raspberry?",
    "Why is the sky blue?"
]

for question in test_questions:
    print(f"\n Testing: {question}")
    final_answer = budget_forcing(question, model="llama3.2:1b")
    print("Final Answer Output:\n", final_answer)
    print("=" * 50)
