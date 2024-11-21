import os

def get_file_paths(base_dir):
    train_files = []
    test_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.off'):
                full_path = os.path.join(root, file)
                if '/train/' in full_path:
                    train_files.append(full_path)
                elif '/test/' in full_path:
                    test_files.append(full_path)
    
    return train_files, test_files

# Usage
base_dir = "ModelNet40"
train_files, test_files = get_file_paths(base_dir)

print("Train files:", len(train_files))
print("Test files:", len(test_files))