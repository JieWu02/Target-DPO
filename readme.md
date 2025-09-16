# target_dpo Setup & Data Format
This repo offers an implementation for our paper "Teaching Your Models to Understand Code via Focal Preference Alignment" (EMNLP 2025 Main Conference).

## Environment Setup
```bash
conda create -n target_dpo python=3.11
conda activate target_dpo
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

## Install Alignment Handbook
```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook/
python -m pip install .
pip install wandb
pip install "numpy<2"
cd ..
```

## Launch Training
```bash
bash train.sh
```

## Input JSON Example
An entry in the input JSON includes the `prompt`, `chosen`, and `rejected` keys, each stored as a string:

```json
{
    "prompt": "**Problem Statement:**\n  \n  You are given a CSV file that contains a list of mathematical expressions in each row. Your task is to implement a function that reads this CSV file, evaluates each mathematical expression, and returns the results as a list of floats. If any expression is invalid, the function should handle the exception by returning a custom error message for that specific expression.\n  \n  The CSV file will have the following format:\n  ```\n  expression\n  \"2+3*5\"\n  \"10/0\"\n  \"(4+5)*7\"\n  \"sqrt(16)\"\n  \"invalid_expression\"\n  ```\n  \n  **Function Requirements:**\n  1. The function should read the CSV file.\n  2. Evaluate each mathematical expression.\n  3. Return a list of results, where each result is either the evaluated value (as a float) or an error message (as a string) if the expression is invalid.\n  \n  **Constraints:**\n  - The expressions will be provided as strings in the CSV file.\n  - The expressions may include basic arithmetic operations (`+`, `-`, `*`, `/`), parentheses, and functions like `sqrt()`.\n  - Handle invalid expressions gracefully with custom error messages.\n  - Assume the CSV file is well-formed, and each row contains only one expression.\n  \n  **Edge Cases:**\n  - Division by zero.\n  - Invalid mathematical functions.\n  - Syntax errors in the expression.\nFunction signature: def evaluate_expressions_from_csv(file_path: str) -> list:\nExamples:\n**Example 1:**\n  - Input: CSV file content:\n    ```\n    expression\n    \"2+3*5\"\n    \"10/0\"\n    \"(4+5)*7\"\n    \"sqrt(16)\"\n    \"invalid_expression\"\n    ```\n  - Output: [17.0, \"Error: Division by zero\", 63.0, 4.0, \"Error: Invalid expression\"]\n\n  **Example 2:**\n  - Input: CSV file content:\n    ```\n    expression\n    \"3.5+2.5\"\n    \"100-50*2\"\n    \"sqrt(25) + 5\"\n    \"invalid*expression\"\n    ```\n  - Output: [6.0, 0.0, 10.0, \"Error: Invalid expression\"]\n\n  **Example 3 (Edge Case - Empty Expression):**\n  - Input: CSV file content:\n    ```\n    expression\n    \"\"\n    ```\n  - Output: [\"Error: Invalid expression\"]\n\n  **Example 4 (Edge Case - Complex Nested Expressions):**\n  - Input: CSV file content:\n    ```\n    expression\n    \"(2*(3+(4/2)))-5\"\n    \"(2*(3+(4/0)))-5\"\n    ```\n  - Output: [5.0, \"Error: Division by zero\"]\nTags:\nread CSV, custom exception handling, mathematical_operations",
    "chosen": "```python\n# evaluate_expressions.py\nimport csv\nimport math\n\ndef evaluate_expressions_from_csv(file_path: str) -> list:\n    results = []\n    \n    with open(file_path, 'r') as file:\n        reader = csv.reader(file)\n        next(reader)  # Skip the header\n        \n        for row in reader:\n            expression = row[0]\n            try:\n                # Safely evaluate the expression\n                result = eval(expression, {\"__builtins__\": None}, {\"sqrt\": math.sqrt})\n                results.append(float(result))\n            except ZeroDivisionError:\n                results.append(\"Error: Division by zero\")\n            except (SyntaxError, NameError, TypeError):\n                results.append(\"Error: Invalid expression\")\n            except Exception as e:\n                results.append(f\"Error: {str(e)}\")\n    \n    return results\n\n# Test case generation\nif __name__ == \"__main__\":\n    # Generating a test CSV file for demonstration purposes\n    test_csv_content = \"\"\"expression\n\"2+3*5\"\n\"10/0\"\n\"(4+5)*7\"\n\"sqrt(16)\"\n\"invalid_expression\"\n\"\"\"\n    test_file_path = 'test_expressions.csv'\n    with open(test_file_path, 'w') as file:\n        file.write(test_csv_content)\n    \n    # Running the function and printing the results\n    results = evaluate_expressions_from_csv(test_file_path)\n    print(results)\n```",
    "rejected": "```python\n# evaluate_expressions.py\nimport csv\nimport math\n\ndef evaluate_expressions_from_csv(file_path: str) -> list:\n    results = []\n    \n    with open(file_path, 'r') as file:\n        reader = csv.reader(file)\n        next(reader)  # Skip the header\n        \n        for row in reader:\n            expression = row[0]\n            try:\n                # Safely evaluate the expression\n                result = eval(expression, {\"__builtins__\": None}, {\"sqrt\": math.sqrt})\n                results.append(float(result))\n            except ZeroDivisionError:\n                results.append(\"Error: Division by zero\")\n            except (SyntaxError, NameError):\n                results.append(\"Error: Invalid expression\")\n            except Exception as e:\n                results.append(f\"Error: {str(e)}\")\n    \n    return results\n\n# Test case generation\nif __name__ == \"__main__\":\n    # Generating a test CSV file for demonstration purposes\n    test_csv_content = \"\"\"expression\n\"2+3*5\"\n\"10/0\"\n\"(4+5)*7\"\n\"sqrt(16)\"\n\"invalid_expression\"\n\"\"\"\n    test_file_path = 'test_expressions.csv'\n    with open(test_file_path, 'w') as file:\n        file.write(test_csv_content)\n    \n    # Running the function and printing the results\n    results = evaluate_expressions_from_csv(test_file_path)\n    print(results)\n```"
}
```

## Citation
```bibtex
@misc{wu2025iterpreffocalpreferencelearning,
      title={IterPref: Focal Preference Learning for Code Generation via Iterative Debugging}, 
      author={Jie Wu and Haoling Li and Xin Zhang and Jianwen Luo and Yangyu Huang and Ruihang Chu and Yujiu Yang and Scarlett Li},
      year={2025},
      eprint={2503.02783},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.02783}, 
}
```
