import os
from pathlib import Path

def generate_context():
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_file = root_dir / 'neural_trader_context.txt'
    
    # Directories to completely ignore
    ignore_dirs = {'.git', '.venv', '__pycache__', '.pytest_cache', 'models', 'logs'}
    # Extensions to read code from
    target_exts = {'.py', '.md', '.json', '.yaml', '.toml'}
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("=========================================\n")
        out.write("NEURALTRADER SYSTEM CONTEXT DUMP\n")
        out.write("=========================================\n\n")
        
        # 1. Generate Directory Tree
        out.write("### 1. DIRECTORY TREE ###\n")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            level = str(root).replace(str(root_dir), '').count(os.sep)
            indent = ' ' * 4 * level
            out.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for f in sorted(files):
                if not f.endswith(('.parquet', '.pkl', '.csv', '.db')): # Ignore massive data files
                    out.write(f"{subindent}{f}\n")
        
        out.write("\n=========================================\n\n")
        
        # 2. Extract Core Code
        out.write("### 2. CORE SOURCE CODE ###\n")
        
        # Specific files I need to see to fix the zombie trades and email wrapper
        critical_files = [
            'main_orchestrator_ist.py',
            'core/notifier.py',
            'core/live_ranker.py',
            'scripts/report_generator.py',
            'docs/ARCHITECTURE.md',
            'data/portfolio.json'
        ]
        
        for file_path in critical_files:
            full_path = root_dir / file_path
            out.write(f"\n--- FILE: {file_path} ---\n")
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"[ERROR READING FILE: {e}]\n")
            else:
                out.write("[FILE NOT FOUND]\n")

if __name__ == "__main__":
    generate_context()
    print("Context dump complete. File saved to neural_trader_context.txt")