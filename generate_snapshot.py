import os

# Configuration
OUTPUT_FILE = 'project_snapshot.txt'

# Only include these file extensions (The "Source Code" Filter)
INCLUDED_EXTENSIONS = {'.py', '.md', '.json', '.yaml', '.yml', '.bat', '.sh', '.txt'}

# Strictly ignore these directories
EXCLUDED_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.idea', '.vscode', 
    'data', 'logs', 'reports', 'models', 'images', 'archive', 'backup', 
    'notebooks', '.windsurf', '.venv', 'node_modules', 'dist', 'build'
}

# Strictly ignore these specific files (even if they have the right extension)
EXCLUDED_FILES = {
    'project_snapshot.txt', 'requirements.txt', 'LICENSE', 
    '.gitignore', 'package-lock.json', 'NEURAL_TRADER_BRAIN_DUMP.txt'
}

# Additional size limits for ultra-compact snapshot
MAX_FILE_SIZE = 50000  # 50KB per file max
MAX_TOTAL_FILES = 500  # Maximum number of files to include

def is_text_file(filename):
    """Check if file has a valid source code extension."""
    return any(filename.endswith(ext) for ext in INCLUDED_EXTENSIONS)

def generate_snapshot():
    project_root = os.getcwd()
    files_processed = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Write Header
        outfile.write(f"PROJECT SNAPSHOT (COMPACT)\n")
        outfile.write(f"==========================\n\n")
        
        # 1. Write Directory Tree (Lightweight)
        outfile.write("DIRECTORY STRUCTURE:\n")
        for root, dirs, files in os.walk(project_root):
            # Filter directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            
            level = root.replace(project_root, '').count(os.sep)
            indent = ' ' * 4 * level
            outfile.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if is_text_file(f) and f not in EXCLUDED_FILES:
                    outfile.write(f"{subindent}{f}\n")
        
        outfile.write("\n" + "="*50 + "\n\n")
        
        # 2. Write File Contents (Source Code Only - Compact)
        for root, dirs, files in os.walk(project_root):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            
            for file in files:
                if files_processed >= MAX_TOTAL_FILES:
                    outfile.write(f"\n[REACHED LIMIT: {MAX_TOTAL_FILES} files processed]\n")
                    break
                    
                if is_text_file(file) and file not in EXCLUDED_FILES:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, project_root)
                    
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write("-" * len(f"FILE: {rel_path}") + "\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()
                            
                            # Apply size limits
                            if len(content) > MAX_FILE_SIZE:
                                truncated_content = content[:MAX_FILE_SIZE]
                                outfile.write(truncated_content)
                                outfile.write(f"\n\n[TRUNCATED: File too large ({len(content)} chars, showing first {MAX_FILE_SIZE})]\n")
                            else:
                                outfile.write(content)
                            
                            files_processed += 1
                            
                    except Exception as e:
                        outfile.write(f"[ERROR READING FILE: {e}]\n")
                    
                    outfile.write("\n\n" + "="*50 + "\n\n")
            
            if files_processed >= MAX_TOTAL_FILES:
                break

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Compact Snapshot generated: {OUTPUT_FILE}")
    print(f"📊 Size: {file_size_mb:.2f} MB")
    print(f"📁 Files processed: {files_processed}/{MAX_TOTAL_FILES}")
    
    if file_size_mb > 100:
        print(f"⚠️  WARNING: File is still over 100MB. Consider reducing MAX_FILE_SIZE or MAX_TOTAL_FILES")

if __name__ == "__main__":
    generate_snapshot()