import os
import sys
from pathlib import Path

def stringify_directory():
    """Read all files in the current directory and write their contents to a single output file."""
    current_dir = Path(".")
    output_file = "codebase_contents.txt"
    
    # Get all files in the current directory
    files = [f for f in current_dir.iterdir() if f.is_file() and f.name != output_file and f.name != "stringify_llm.py"]
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("# Codebase Contents\n\n")
        
        for file_path in sorted(files):
            try:
                # Skip binary files or very large files
                if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                    out_f.write(f"<file path=\"{file_path}\">\nFile too large to include (> 10MB)\n</file>\n\n")
                    continue
                
                # Try to read the file as text
                with open(file_path, "r", encoding="utf-8") as in_f:
                    content = in_f.read()
                
                # Write the file name and content with clear XML tags
                out_f.write(f"<file path=\"{file_path}\">\n")
                out_f.write(content)
                # Ensure there's a newline at the end of the file
                if not content.endswith("\n"):
                    out_f.write("\n")
                out_f.write("</file>\n\n")
                
                print(f"Processed: {file_path}")
                
            except UnicodeDecodeError:
                # Handle binary files
                out_f.write(f"<file path=\"{file_path}\">\nBinary file - content not included\n</file>\n\n")
                print(f"Skipped binary file: {file_path}")
                
            except Exception as e:
                # Handle any other errors
                out_f.write(f"<file path=\"{file_path}\">\nError reading file: {str(e)}\n</file>\n\n")
                print(f"Error processing {file_path}: {e}")
    
    print(f"\nCompleted! All file contents written to {output_file}")
    print(f"Processed {len(files)} files")

if __name__ == "__main__":
    stringify_directory()