import gradio as gr
import wikipedia
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a more powerful model (FLAN-T5-Large)
MODEL_NAME = "google/flan-t5-xl"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# Tool 1: Wikipedia Search
def wikipedia_search(query):
    """Search Wikipedia and return a summary."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=3)
    except wikipedia.exceptions.PageError:
        return "Page not found. Try a different search term."
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 2: Python Calculator
def calculator(expression):
    """Evaluate a Python mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define available tools
TOOLS = {
    "Wikipedia": {
        "function": wikipedia_search,
        "description": "Useful for finding factual information about people, places, events, and general knowledge."
    },
    "Calculator": {
        "function": calculator,
        "description": "Useful for mathematical calculations. Input should be a valid Python expression."
    }
}

def generate_text(prompt, max_length=512):
    """Generate text from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=1,
        do_sample=False,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_react_prompt(question, tools, history=""):
    """Create a prompt with strong few-shot examples."""
    prompt = f"""You must use tools to answer questions. Always respond with Action and Action Input.

Example 1:
Question: What is 100 times 50?
Action: Calculator
Action Input: 100 * 50

Example 2:
Question: Who is Barack Obama?
Action: Wikipedia
Action Input: Barack Obama

Example 3:
Question: What is the capital of Germany?
Action: Wikipedia
Action Input: Germany capital

{history}Question: {question}
Action:"""
    
    return prompt

def run_agent(question, max_iterations=5):
    """
    Run the ReAct agent loop and return the reasoning trace and final answer.
    """
    history = ""
    observations = []
    trace = []
    
    trace.append(f"🤔 **Question:** {question}\n")
    
    for iteration in range(max_iterations):
        trace.append(f"\n### Iteration {iteration + 1}\n")
        
        # Generate the prompt
        prompt = create_react_prompt(question, TOOLS, history)
        
        # Get model response
        response = generate_text(prompt, max_length=64)
        trace.append(f"**Model Output:** {response}\n")
        
        # Parse the response - extract action name
        action_name = None
        action_input = None
        
        # Try to find tool names in response
        for tool_name in TOOLS.keys():
            if tool_name.lower() in response.lower():
                action_name = tool_name
                parts = response.lower().split(tool_name.lower())
                if len(parts) > 1:
                    action_input = parts[1].strip()
                    action_input = re.sub(r'^[:\s]+', '', action_input)
                    action_input = action_input.split('\n')[0].strip()
                break
        
        # If we found an action, execute it
        if action_name and action_input:
            trace.append(f"🔧 **Action:** {action_name}\n")
            trace.append(f"📥 **Action Input:** {action_input}\n")
            
            observation = TOOLS[action_name]['function'](action_input)
            observations.append(observation)
            
            trace.append(f"👁️ **Observation:** {observation[:200]}...\n")
            
            history += f"Observation: {observation}\n\n"
            
            # After getting observations, check if we can answer
            if len(observations) >= 2:
                trace.append(f"\n✅ **Final Answer:** {observations[-1]}\n")
                return "\n".join(trace), observations[-1]
        
        else:
            # Fallback logic
            trace.append("⚠️ Using fallback logic...\n")
            
            if not observations:
                # First iteration - search Wikipedia
                search_terms = question.lower()
                search_terms = re.sub(r'\b(what|is|the|of|squared?|multiplied?|times?|plus?)\b', '', search_terms)
                search_terms = search_terms.strip()
                
                trace.append(f"🔧 **Fallback Action:** Wikipedia\n")
                trace.append(f"📥 **Action Input:** {search_terms}\n")
                
                observation = wikipedia_search(search_terms)
                observations.append(observation)
                
                trace.append(f"👁️ **Observation:** {observation[:200]}...\n")
                
                history += f"Observation: {observation}\n\n"
            
            elif len(observations) == 1:
                # Second iteration - do calculation
                # Extract numbers from observation
                large_numbers = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', observations[0])
                
                if not large_numbers:
                    million_match = re.search(r'(\d+(?:\.\d+)?)\s*million', observations[0], re.IGNORECASE)
                    billion_match = re.search(r'(\d+(?:\.\d+)?)\s*billion', observations[0], re.IGNORECASE)
                    
                    if million_match:
                        num_str = million_match.group(1)
                        num = float(num_str) * 1_000_000
                        large_numbers = [str(int(num))]
                    elif billion_match:
                        num_str = billion_match.group(1)
                        num = float(num_str) * 1_000_000_000
                        large_numbers = [str(int(num))]
                
                if not large_numbers:
                    all_numbers = re.findall(r'\b\d+\b', observations[0])
                    large_numbers = [n for n in all_numbers if int(n) > 1000]
                
                if large_numbers and "squared" in question.lower():
                    num = large_numbers[0].replace(',', '')
                    calc_expr = f"{num} ** 2"
                    
                    trace.append(f"🔧 **Fallback Action:** Calculator\n")
                    trace.append(f"📥 **Action Input:** {calc_expr}\n")
                    
                    result = calculator(calc_expr)
                    
                    trace.append(f"👁️ **Observation:** {result}\n")
                    trace.append(f"\n✅ **Final Answer:** {result}\n")
                    
                    return "\n".join(trace), result
                
                elif large_numbers and ("times" in question.lower() or "multiplied" in question.lower()):
                    multiplier = re.findall(r'\b(\d+)\b', question)
                    if multiplier:
                        num = large_numbers[0].replace(',', '')
                        calc_expr = f"{num} * {multiplier[-1]}"
                        
                        trace.append(f"🔧 **Fallback Action:** Calculator\n")
                        trace.append(f"📥 **Action Input:** {calc_expr}\n")
                        
                        result = calculator(calc_expr)
                        
                        trace.append(f"👁️ **Observation:** {result}\n")
                        trace.append(f"\n✅ **Final Answer:** {result}\n")
                        
                        return "\n".join(trace), result
                else:
                    return "\n".join(trace), "Could not extract number for calculation."
    
    return "\n".join(trace), "Unable to determine answer."

def process_question(question):
    """Process a question and return the reasoning trace and answer."""
    if not question.strip():
        return "Please enter a question.", ""
    
    trace, answer = run_agent(question, max_iterations=5)
    return trace, answer

# Create Gradio interface
with gr.Blocks(title="ReAct Agent - Reasoning + Acting", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 ReAct Agent: Reasoning + Acting
    
    This agent can answer questions by using tools (Wikipedia and Calculator).
    
    **How it works:**
    1. The agent reads your question
    2. It decides which tool to use (Wikipedia for facts, Calculator for math)
    3. It executes the tool and observes the result
    4. It repeats until it has enough information to answer
    
    **Try these examples:**
    - "What is the population of France squared?"
    - "What is the area of Germany multiplied by 3?"
    - "What is 100 plus the population of Tokyo in millions?"
    
    **Model:** google/flan-t5-large (770M parameters)
    """)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What is the population of France squared?",
                lines=3
            )
            submit_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Final Answer",
                lines=3,
                interactive=False
            )
    
    with gr.Row():
        trace_output = gr.Markdown(
            label="Reasoning Trace",
            value="The agent's step-by-step reasoning will appear here..."
        )
    
    # Examples
    gr.Examples(
        examples=[
            ["What is the population of France squared?"],
            ["What is the area of Germany multiplied by 3?"],
            ["What is 100 plus the population of Tokyo in millions?"],
            ["Who was Albert Einstein and what year was he born?"],
        ],
        inputs=question_input
    )
    
    # Connect the button
    submit_btn.click(
        fn=process_question,
        inputs=question_input,
        outputs=[trace_output, answer_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
