import re
import os

def extract_python_code(text, name):
    """
    Extracts and prints blocks of Python code from a given text that are delimited by
    ```python and ```
    """
    # Regular expression to find all occurrences of Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)

    GEN_CODE_DIR = os.getenv("GEN_CODE_DIR")
    assert GEN_CODE_DIR is not None, "Please set GEN_CODE_DIR env var"
    os.makedirs(GEN_CODE_DIR, exist_ok=True)

    for i, block in enumerate(code_blocks, start=1):
        # Strip leading/trailing whitespace and maintain internal formatting
        formatted_block = block.strip()
        # Define file path for each code block
        filename = GEN_CODE_DIR + "/{}_{}.py".format(name, i)
        print("generated code filename", filename, "\n")

        with open(filename, 'w') as file:
            file.write(formatted_block)
        print(f"Code Block {i} saved to {filename}\n")
        return filename

def extract_error_messages(stderr_output):
    """
    Extracts error messages from stderr output.
    """
    # Split the stderr output into lines
    lines = stderr_output.split('\n')
    
    # Initialize a list to store error messages
    error_messages = []

    # Extract lines that contain error messages
    for i, line in enumerate(lines):
        if 'Traceback (most recent call last):' in line:
            # Start of a new traceback, find the next line starting with 'File'
            for j in range(i+1, len(lines)):
                if lines[j].strip().startswith('File'):
                    # Add lines until 'AttributeError' or other errors are encountered
                    error_detail = lines[j].strip()
                    k = j + 1
                    while k < len(lines) and not lines[k].strip().startswith('File'):
                        error_detail += '\n' + lines[k].strip()
                        k += 1
                    error_messages.append(error_detail)
                    break
    
    return error_messages
