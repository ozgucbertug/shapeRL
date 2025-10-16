import os

def should_ignore(path, ignore_list):
    for ignore in ignore_list:
        if ignore in path:
            return True
    return False

def find_python_files(directory, ignore_list):
    python_files = []
    for root, dirs, files in os.walk(directory):
        if should_ignore(root, ignore_list):
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not should_ignore(file_path, ignore_list):
                    python_files.append(file_path)
    return python_files

def save_to_file(file_list, output_file):
    with open(output_file, 'w') as f:
        for file in file_list:
            with open(file, 'r') as infile:
                f.write(f"# {file}\n")
                f.write('"""python\n')
                f.write(infile.read())
                f.write('\n"""\n\n')

if __name__ == "__main__":
    directory_to_search = "."  # Change this to the directory you want to search
    ignore_list = ['out/', 'util/', '__init__']  # Add more directories to ignore if needed
    output_file = 'util/out.txt'

    python_files = find_python_files(directory_to_search, ignore_list)
    # print(python_files)
    save_to_file(python_files, output_file)