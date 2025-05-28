from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import random
import re

# Load model & tokenizer TinyLLaMA (1.1B)
print("🔄 Loading model...")
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# System prompt to guide assistant personality
system_prompt = (
    "You are Math Assistant, a friendly AI that helps students understand math. "
    "You explain using simple language and step-by-step examples.\n"
)

# Save chat history to a file (append mode)
def save_to_log(text, filename="chat_history.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Evaluate simple math expressions safely
def try_eval_math(expr):
    # Remove any = or ? or spaces
    expr = expr.strip().rstrip("=? ").strip()
    
    # Allow only numbers, operators, and parentheses
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        return None
    
    try:
        # Evaluate the expression
        result = eval(expr)
        return str(result)
    except:
        return None

# Expanded quiz with multiple choice
def run_quiz():
    questions = [
        {"q": "5 + 3?", "a": "8", "choices": None},
        {"q": "9 - 4?", "a": "5", "choices": None},
        {"q": "6 x 7?", "a": "42", "choices": None},
        {"q": "16 / 4?", "a": "4", "choices": None},
        {"q": "3 ^ 2?", "a": "9", "choices": None},
        # Multiple choice questions
        {"q": "What is 10 + 5?", "a": "15", "choices": ["10", "15", "20", "25"]},
        {"q": "What is 12 / 3?", "a": "4", "choices": ["3", "4", "6", "9"]},
        {"q": "What is 7 x 6?", "a": "42", "choices": ["36", "40", "42", "48"]},
    ]
    q = random.choice(questions)
    if q["choices"]:
        # Present multiple choice
        print("🧠 Quiz: " + q["q"])
        for i, choice in enumerate(q["choices"], 1):
            print(f"  {i}. {choice}")
        user_answer = input("Your answer (number): ").strip()
        # Validate input number and map to choice string
        try:
            user_choice = q["choices"][int(user_answer) - 1]
        except:
            print("❌ Invalid input.\n")
            return
        if user_choice == q["a"]:
            print("✅ Correct!\n")
        else:
            print(f"❌ Incorrect. The right answer is {q['a']}.\n")
    else:
        # Open question
        user_answer = input(f"🧠 Quiz: {q['q']} ")
        if user_answer.strip() == q["a"]:
            print("✅ Correct!\n")
        else:
            print(f"❌ Incorrect. The right answer is {q['a']}.\n")

# Main chat loop
def chat():
    print("📘 Welcome to Math Assistant!")
    print("Type 'quiz' for a math quiz, or 'exit', 'bye', 'quit' to exit the program.\n")

    # Ask for custom filename for chat log
    filename = input("Enter chat log filename (default: chat_history.txt): ").strip()
    if filename == "":
        filename = "chat_history.txt"

    history = ""
    while True:
        user_input = input("👤 You: ").strip()

        # Exit condition
        if user_input.lower() in ["exit", "bye", "quit"]:
            print(f"👋 Goodbye! Your chat history is saved in '{filename}'.")
            break

        # Run quiz if requested
        if user_input.lower() == "quiz":
            run_quiz()
            continue

        # Try to evaluate math expression first (fast response)
        math_answer = try_eval_math(user_input)
        if math_answer is not None:
            print(f"🤖 Math Assistant: The answer is {math_answer}\n")
            # Save chat log
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            save_to_log(f"{timestamp} User: {user_input}", filename)
            save_to_log(f"{timestamp} Assistant: The answer is {math_answer}", filename)
            # Add to history for context
            history += f"User: {user_input}\nAssistant: The answer is {math_answer}\n"
            continue

        # Build prompt including system message and chat history
        prompt = system_prompt + history + f"User: {user_input}\nAssistant:"

        # Tokenize input and move to correct device (CPU/GPU)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response from model
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode output tokens to string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's answer after the last 'Assistant:'
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print(f"🤖 Math Assistant: {response}\n")

        # Update chat history for context in future responses
        history += f"User: {user_input}\nAssistant: {response}\n"

        # Save current turn to log file with timestamp
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        save_to_log(f"{timestamp} User: {user_input}", filename)
        save_to_log(f"{timestamp} Assistant: {response}", filename)

if __name__ == "__main__":
    chat()
