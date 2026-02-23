import os
import shutil
from pathlib import Path

def burn_the_trash():
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Target Directories to completely eradicate
    trash_dirs = [
        root / 'analysis_results',
        root / 'scripts' / 'archive',
        root / 'scripts' / 'setup'
    ]
    
    # 2. Specific root clutter files
    trash_files = [
        root / 'calculate_cagr.py',
        root / 'debug_sentiment_data.json'
    ]
    
    # 3. Purge old markdown summaries in scripts
    for md_file in (root / 'scripts').glob('*_summary.md'):
        trash_files.append(md_file)
    
    trash_files.append(root / 'scripts' / 'pipeline_rebuild_report.md')
    trash_files.append(root / 'scripts' / 'standardization_report.md')
    trash_files.append(root / 'scripts' / 'task_setup_report.md')
    
    # Execute Deletions
    for d in trash_dirs:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            print(f"[DELETED DIR] {d.name}/")
            
    for f in trash_files:
        if f.exists() and f.is_file():
            f.unlink()
            print(f"[DELETED FILE] {f.name}")

if __name__ == '__main__':
    burn_the_trash()
    print('Repo Zen Achieved.')
