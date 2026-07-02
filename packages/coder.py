import os
import re
import json
import subprocess
import sys
from groq import Groq

# ---------------- CONFIGURATION ----------------
GROQ_API_KEY = "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"
MEMORY_FILE = "agent_memory.json"
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an elite, autonomous Python coding agent. 
You have access to the user's local file system to explore, read, write, and test code.

When you need to interact with the system, use the following exact XML formats. 
IMPORTANT: Use ONLY ONE tool per response. Wait for the system to reply with the result before proceeding.

1. List files in a directory to understand the codebase:
<list_dir>./</list_dir>

2. Read a file's contents:
<read_file>filename.py</read_file>

3. Write or overwrite a file:
<write_file path="filename.py">
# your code here
</write_file>

4. Execute a python script to test it:
<execute>filename.py</execute>

If you encounter an error during execution, read the code, fix the logic, and write it again.
If the user asks a normal question, just reply normally without tags.
"""

# ---------------- TOOL FUNCTIONS ----------------
def list_dir(path):
    try:
        files = os.listdir(path)
        return f"Directory contents of '{path}':\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {e}"

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f"--- Contents of {path} ---\n" + f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path, content):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        return f"Success: Wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def execute_file(path):
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout: output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr: output += f"STDERR:\n{result.stderr}\n"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (possible infinite loop)."
    except Exception as e:
        return f"Error executing file: {e}"

# ---------------- MEMORY MANAGEMENT ----------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def save_memory(memory):
    # Keep memory manageable: keep system prompt + last 20 messages
    if len(memory) > 21:
        memory = [memory[0]] + memory[-20:]
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=4)

# ---------------- AGENT LOOP ----------------
def process_agent_response(response_text):
    """Parses the text for XML tags and executes the requested tool."""
    # Check for write
    write_match = re.search(r'<write_file path="(.*?)">(.*?)</write_file>', response_text, re.DOTALL)
    if write_match:
        return write_file(write_match.group(1), write_match.group(2))
    
    # Check for read
    read_match = re.search(r'<read_file>(.*?)</read_file>', response_text)
    if read_match:
        return read_file(read_match.group(1))
    
    # Check for execute
    exec_match = re.search(r'<execute>(.*?)</execute>', response_text)
    if exec_match:
        return execute_file(exec_match.group(1))
    
    # Check for list_dir
    dir_match = re.search(r'<list_dir>(.*?)</list_dir>', response_text)
    if dir_match:
        return list_dir(dir_match.group(1))

    return None # No tool used

def chat_loop():
    print("🤖 Agent CLI initialized. Type 'quit' or 'exit' to stop.")
    print("💾 Memory loaded. The agent remembers previous conversations.")
    memory = load_memory()
    
    while True:
        try:
            user_input = input("\n[null@archlinux ~]$ ")
            if user_input.lower() in ['quit', 'exit']:
                print("Saving memory and exiting...")
                save_memory(memory)
                break
            if not user_input.strip(): continue

            memory.append({"role": "user", "content": user_input})
            
            # Agentic Loop (loops automatically if a tool is used)
            agent_working = True
            while agent_working:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=memory,
                    temperature=0.2,
                    stream=True
                )
                
                print("\n🤖 Agent: ", end="", flush=True)
                full_response = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
                print("\n")
                
                memory.append({"role": "assistant", "content": full_response})
                save_memory(memory)

                # Check if the agent wants to use a tool
                tool_result = process_agent_response(full_response)
                
                if tool_result:
                    print(f"⚙️  [System execution result]:\n{tool_result}")
                    # Feed the result back into the LLM
                    memory.append({"role": "user", "content": f"Tool Result:\n{tool_result}"})
                    # Loop continues, LLM will read the result and respond again
                else:
                    # No tool used, agent is done with this thought
                    agent_working = False
                    
        except KeyboardInterrupt:
            print("\nSaving memory and exiting...")
            save_memory(memory)
            break
        except Exception as e:
            print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    chat_loop()