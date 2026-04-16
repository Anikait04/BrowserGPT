import os

# Folder(s) to ignore
EXCLUDE = {'.venv', '.git', '__pycache__', '.vscode'}

# Output file
OUTPUT_FILE = 'tree.txt'

def write_tree(path, prefix='', to_file=True):
    items = sorted(os.listdir(path))
    items = [i for i in items if i not in EXCLUDE]

    for index, item in enumerate(items):
        full_path = os.path.join(path, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        line = f"{prefix}{connector}{item}"

        if to_file:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        else:
            print(line)

        # Recurse if directory
        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if index == len(items) - 1 else "│   ")
            write_tree(full_path, new_prefix, to_file)

user_input = int(input("Enter\n1 to Show in Terminal\n2 to Save in File: "))

if user_input == 1:
    print("BrowserGPT")
    write_tree(os.getcwd(), to_file=False)
else:
    open(OUTPUT_FILE, 'w').close()
    write_tree(os.getcwd(), to_file=True)
    print(f"Tree structure saved to {OUTPUT_FILE}")