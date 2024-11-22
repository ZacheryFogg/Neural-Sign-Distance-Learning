import os

def repair_off_files(dataset_dir):
    """
    Repairs .off files in the given dataset directory by ensuring the first line starts with 'OFF'
    followed by a newline. Fixes cases like 'OFF700 664 0' to 'OFF\n700 664 0'.
    
    Args:
        dataset_dir (str): The base directory containing the .off files to be repaired.
    """
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".off"):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Check if the first line needs repair
                if lines and not lines[0].startswith("OFF\n"):
                    first_line = lines[0].strip()
                    
                    # If it begins with "OFF", split it into "OFF" and the rest
                    if first_line.startswith("OFF"):
                        rest_of_line = first_line[3:].strip()  # Extract what's after "OFF"
                        lines[0] = "OFF\n"
                        
                        # Add the rest of the original first line to the next line
                        if rest_of_line:
                            lines.insert(1, rest_of_line + "\n")
                    
                    # Write the repaired file back
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                print(f"Checked and repaired (if needed): {file_path}")

# Example usage
dataset_dir = "../ModelNet40"
repair_off_files(dataset_dir)
