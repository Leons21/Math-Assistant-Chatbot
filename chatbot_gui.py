from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import re
from datetime import datetime

# Load model & tokenizer
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# System prompt 
system_prompt = (
    "You are Math Assistant, a friendly AI that helps students understand math. "
    "You explain using simple language and step-by-step examples.\n"
)

def convert_text_to_math(expr):
    expr = expr.lower()
    expr = expr.replace("squared", "**2")
    expr = expr.replace("cubed", "**3")
    expr = re.sub(r"to the power of (\d+)", r"**\1", expr)
    expr = expr.replace("x", "*")  # Convert "2 x 2" to "2 * 2"
    return expr

def try_eval_math(expr):
    expr = convert_text_to_math(expr.strip().rstrip("=? ").strip())
    if not re.fullmatch(r"[0-9+\-*/(). **]+", expr):
        return None
    try:
        return str(eval(expr))
    except:
        return None

# Chat log file / history
def save_to_log(text, filename="chat_history.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Gradio chat
def respond(user_input, history, quiz_state):
    # Quiz mode | check the answer
    if quiz_state and quiz_state.get("active", False):
        correct_answer = quiz_state.get("answer")
        if user_input.strip().lower() == correct_answer.strip().lower():
            bot_response = "✅ Correct! Well done."
        else:
            bot_response = f"❌ Incorrect. The correct answer is: {correct_answer}"
        quiz_state["active"] = False
        history.append((user_input, bot_response))
    
    # Quiz mode
    elif "quiz" in user_input.lower():
        quiz_question = "What is 12 × 7?"
        correct_answer = "84"
        quiz_state = {"active": True, "answer": correct_answer}
        bot_response = quiz_question
        history.append((user_input, bot_response))

    # Answer Interactions
    else:
        math_ans = try_eval_math(user_input)
        if math_ans is not None:
            bot_response = f"The answer is {math_ans}"
        else:
            prompt = system_prompt
            for h in history:
                prompt += f"User: {h[0]}\nAssistant: {h[1]}\n"
            prompt += f"User: {user_input}\nAssistant:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_response = full.split("Assistant:")[-1].strip()

        history.append((user_input, bot_response))

    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    save_to_log(f"{ts} You: {user_input}")
    save_to_log(f"{ts} Bot: {bot_response}")
    return history, history, quiz_state

# UI Gradio setup
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Math Assistant Chatbot + Quiz")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a math question or type 'quiz'...")
    clear = gr.Button("Clear Chat")
    state = gr.State([])
    quiz_state = gr.State({})

    msg.submit(respond, [msg, state, quiz_state], [chatbot, state, quiz_state])
    clear.click(lambda: ([], [], {}), None, [chatbot, state, quiz_state])

if __name__ == "__main__":
    demo.launch()