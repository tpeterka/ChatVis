# ChatVis agent to run a single test case in the ChatVis benchmark

# Import required modules

# --- Standard library ---
import os
import sys
import json
import pickle
import subprocess
import numpy as np

# --- Third-party libraries ---
import faiss
from sentence_transformers import SentenceTransformer

# --- Local utilities ---
from Utility.extract_utils import extract_python_code, extract_error_messages
from Utility.llm_client import LLMClient

# Path to pvpython
PATH_TO_PVPYTHON = os.getenv("PATH_TO_PVPYTHON")
assert PATH_TO_PVPYTHON is not None, "Please set PATH_TO_PVPYTHON env var"
path_to_pvpython = PATH_TO_PVPYTHON + ":$PATH"  
print("path to pvpython", PATH_TO_PVPYTHON)
os.environ["PATH"] += os.pathsep + path_to_pvpython

# LLM provider/model/credentials, set through environment variables:
# LLM_PROVIDER, LLM_MODEL, LLM_BASEURL + provider API key (see Utility/llm_client.py)
llm = LLMClient()
print("LLM provider:", llm.provider, "model:", llm.model)

# Read and parse the JSON file
json_file_path = "operations.json"
with open(json_file_path, "r") as file:
    operations_json = json.load(file)

# Load a smaller Sentence Transformer model for efficiency
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embeddings safely (one by one)
def get_embedding(text):
    return model.encode(text, convert_to_numpy=True).astype(np.float32)

# Initialize FAISS index
d = model.get_embedding_dimension()  # Get correct embedding dimension
index = faiss.IndexFlatL2(d)  # Use L2 distance metric for FAISS

# Generate and add embeddings for each operation
metadata_lookup = []
for op in operations_json:
    text = op["name"] + " " + op["description"] + " " + op["code_snippet"]
    embedding = get_embedding(text).reshape(1, -1)  # Reshape for FAISS
    index.add(embedding)  # pyright: ignore[reportCallIssue]
    metadata_lookup.append(op)  # Store the original entry

# Save the FAISS index to disk
faiss.write_index(index, "paraview_operations_faiss.index")

with open("metadata_lookup.pkl", "wb") as f:
    pickle.dump(metadata_lookup, f)
print("All ParaView operations stored in FAISS database.")

# Load FAISS index (ensure it exists)
index = faiss.read_index("paraview_operations_faiss.index")
with open("metadata_lookup.pkl", "rb") as f:
    metadata_lookup = pickle.load(f)

# Search similar operations
def search_similar_operation(query_text, top_k=5):
    # Generate query embedding
    query_embedding = get_embedding(query_text).reshape(1, -1)

    # Ensure there are enough vectors in the FAISS index before searching
    total_vectors = index.ntotal
    if total_vectors == 0:
        print("Error: FAISS index is empty! No vectors found.")
        return []

    # Get nearest neighbors from FAISS
    top_k = min(top_k, total_vectors)  # Ensure we don't exceed available entries
    distances, indices = index.search(query_embedding, top_k)  # pyright: ignore[reportCallIssue]

    matches = []
    for idx in indices[0]:  # I is shape (1, k)
        match = metadata_lookup[idx]
        matches.append(match)

    return matches

# Process prompt
def process_prompt(prompt: str):
    """
    Process a text prompt by splitting into lines, searching for similar operations,
    and returning unique operations by name.

    Args:
        prompt (str): The input prompt string.

    Returns:
        dict: A dictionary of unique operations keyed by name.
    """
    # Split prompt into non-empty lines
    prompt_lines = [line.strip() for line in prompt.strip().split('\n') if line.strip()]

    # Collect results
    all_results = []
    for line in prompt_lines:
        result = search_similar_operation(line)
        all_results.append(result)

    # Flatten list of lists
    flat_list = [item for sublist in all_results for item in sublist]

    # Deduplicate by name
    unique_ops = {}
    for op in flat_list:
        name = op['name']
        if name not in unique_ops:
            unique_ops[name] = op  # Keep the first occurrence

    return unique_ops

# System prompt
system_prompt =f'''You are a highly accurate code assistant specializing in 3D visualization scripting (e.g., ParaView, VTK). Your task is to read and execute the user's prompt line by line, ensuring that all operations, camera angles, views, rendering, and screenshots are handled correctly.

Execution Rules:
Process the Prompt Line-by-Line

Read and execute each instruction in order without skipping or merging steps.
If an operation depends on a previous step, ensure proper sequencing.
Camera and Viewing Directions

Object Creation and Rendering
Unless the user specifically instructs you to not show a data source, please show any data source after it has been loaded or created.

Apply background settings before rendering.
If a white background is needed for screenshots, ensure it is set before rendering.
Save screenshots immediately after rendering, before moving to the next step.
Ensure filenames or saving locations match the user's intent.

Camera and Viewing Directions
If a specific camera direction or position is given by the user adjust the camera accordingly.
If the user does not specify how to zoom the camera, zoom the camera to fit the active rendered objects as the last operation in the script. Also, zoom the camera to fit the active rendered objects immediately before saving a screenshot. Call ResetCamera() on the render view object so that the camera will be zoomed to fit.
If the user manually specifies a camera zoom level, follow their instructions and do not insert extra calls to 'renderView.ResetCamera();layout = CreateLayout(name='Layout')layout.AssignView(0, renderView)'.

Use provided operation templates as references.
Maintain correct syntax, function calls, and parameters.
Code Quality & Best Practices

Ensure modular, readable, and structured code.
Add comments to explain significant steps.
Avoid redundant operations and ensure compatibility with visualization libraries.
Primary Goal:
Generate a precise, structured, and error-free script that accurately follows the user's instructions, handling camera angles, views, rendering, and screenshots correctly. If any ambiguity exists, infer the most logical approach based on best practices. Follow Example Operations \n{operations_json}'''

# -----  Execute the agent on a single test case -----

def run_single_test_case(test_case_path):
    """
    Execute a single test case from the given path.

    Args:
        test_case_path (str): Path to the test case directory
    """

    python_file_name = '-full-prompt'
    # python_file_name = '-quick-prompt'

    eval_folder = os.getenv("GEN_VIS_DIR")
    assert eval_folder is not None, "Please set GEN_VIS_DIR env var"
    os.makedirs(eval_folder, exist_ok=True)
    print("generated visualizations will be in", eval_folder, "\n")

    # Validate the test case path
    if not os.path.exists(test_case_path):
        print(f"Error: Test case path does not exist: {test_case_path}")
        return

    if not os.path.isdir(test_case_path):
        print(f"Error: Path is not a directory: {test_case_path}")
        return

    task = os.path.basename(test_case_path)
    print("\ntask", task, "in folder", test_case_path, "\n")

    # Check if prompt file exists
    prompt_file = os.path.join(test_case_path, "full_prompt.txt")
    # prompt_file = os.path.join(test_case_path, "quick_prompt.txt")

    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file not found: {prompt_file}")
        return

    with open(prompt_file, "r") as file:
        prompt = file.read()
        print(prompt)

    unique_ops = process_prompt(prompt)
    prompt_new = prompt + f'Follow Example Operations \n{unique_ops}'

    # Call the LLM
    script = llm.chat(system_prompt, prompt_new)

    # print("script:\n", script, "\n")
    # print("task+python_file_name:", task+python_file_name)

    file_path = extract_python_code(script, task+python_file_name)
    cfp = test_case_path
    if file_path:

        command = [PATH_TO_PVPYTHON + "/pvpython", file_path]

        result = subprocess.run(command, capture_output=True, text=True, cwd=cfp)
        errors = extract_error_messages(result.stderr)
        source_folder = cfp
        file_to_copy = task + "-screenshot.png"

        if not errors:
            subprocess.run(["mv", os.path.join(source_folder, file_to_copy), eval_folder])
        else:
            print("Error message is: ", result.stderr)
            print(errors)

        attempts = 0
        while errors and attempts < 5:

            attempts = attempts + 1

            followup_question = f"""
            I tried running the following Python script and encountered an error.

            **Error Message:**
            {errors}

            **Original Script:**
            {script}

            Can you help me fix the issue and provide a corrected version of the script?
            Please make sure the new script runs correctly without errors.
            """

            # Call the LLM again
            script = llm.chat(system_prompt, followup_question)
            file_path = extract_python_code(script, task+python_file_name)

            # Execute the new script with pvpython
            result = subprocess.run(command, capture_output=True, text=True, cwd=cfp)

            # Extract errors from stderr, if any
            errors = extract_error_messages(result.stderr)
            if not errors:
                print("No more errors detected. Script executed successfully.")
                result = subprocess.run(command, capture_output=True, text=True, cwd=cfp)
                print("Error message is: ", result.stderr)
                source_folder = cfp
                file_to_copy = task + "-screenshot.png"
                subprocess.run(["mv", os.path.join(source_folder, file_to_copy), eval_folder])

                break
            else:
                print("Errors detected. Trying again...")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_one.py <test_case_path>")
        print("Example: python run_one.py /path/to/ChatVis_benchmark/test_cases/task1/test_case_1")
        sys.exit(1)

    test_case_path = sys.argv[1]
    run_single_test_case(test_case_path)
    print("completed run_single_test_case", flush=True)
