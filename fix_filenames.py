import os

# Define where your results are
RESULTS_DIR = './results/ringallreduce/sketched_gns'

def rename_files():
    print(f"Scanning {RESULTS_DIR} for legacy filenames...")
    
    count = 0
    
    for root, dirs, files in os.walk(RESULTS_DIR):
        for filename in files:
            if not filename.endswith('.pt'):
                continue
                
            # Remove extension
            name_no_ext = filename[:-3] 
            
            # Split by underscore
            parts = name_no_ext.split('_')
            
            # We need to find where lr_type ends.
            # It could be 'const' (1 part), 'step_decay' (2 parts), or 'exp_decay' (2 parts).
            
            lr_type_end_index = -1
            
            # 1. Check for 'const'
            if 'const' in parts:
                lr_type_end_index = parts.index('const')
            
            # 2. Check for split 'step_decay' or 'exp_decay'
            else:
                for i in range(len(parts) - 1):
                    # Check for ['step', 'decay']
                    if parts[i] == 'step' and parts[i+1] == 'decay':
                        lr_type_end_index = i + 1
                        break
                    # Check for ['exp', 'decay']
                    if parts[i] == 'exp' and parts[i+1] == 'decay':
                        lr_type_end_index = i + 1
                        break
            
            if lr_type_end_index == -1:
                print(f"Could not find lr_type in {filename}, skipping.")
                continue
            
            # The optim field should be immediately after the last part of lr_type
            optim_field_index = lr_type_end_index + 1
            
            # Safety check: does this index exist?
            if optim_field_index >= len(parts):
                print(f"File {filename} ends prematurely after lr_type.")
                continue

            current_val = parts[optim_field_index]
            
            # If it's already a valid optimizer string, we are good. Skip.
            if current_val in ['sgd', 'momentum']:
                continue
            
            # If it looks like a number (the 'steps' argument), we need to insert 'sgd'
            if current_val.isdigit():
                # INSERT 'sgd'
                parts.insert(optim_field_index, 'sgd')
                
                # Reconstruct
                new_name_no_ext = "_".join(parts)
                new_filename = new_name_no_ext + ".pt"
                
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_filename)
                
                print(f"Renaming:\n  OLD: {filename}\n  NEW: {new_filename}")
                os.rename(old_path, new_path)
                count += 1

    print(f"Done. Renamed {count} files.")

if __name__ == "__main__":
    rename_files()