#!/usr/bin/env python
"""
Програмний запуск MLE-STAR для iris датасету.
Використовує ADK API замість інтерактивного CLI.

Автор: Фефелов Ілля Олександрович
"""

import sys
import os
from pathlib import Path

# Додаємо MLE-STAR до шляху
mle_star_root = Path(__file__).parent.parent / "adk-samples" / "python" / "agents" / "machine-learning-engineering"
sys.path.insert(0, str(mle_star_root))

# Змінюємо робочу директорію на MLE-STAR root
os.chdir(mle_star_root)

# Налаштування environment
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = '0'  # Використовуємо ML Dev backend
os.environ['GOOGLE_API_KEY'] = 'AIzaSyChxgm8aM4JHblbMz-152YoU6ULPjWvJg4'
os.environ['ROOT_AGENT_MODEL'] = 'gemini-2.0-flash-lite'

print("="*80)
print("PROGRAMMATIC MLE-STAR RUN: IRIS DATASET")
print("="*80)
print(f"Working directory: {os.getcwd()}")
print(f"Model: {os.environ.get('ROOT_AGENT_MODEL')}")
print("="*80)

try:
    # Імпортуємо агента
    from machine_learning_engineering.agent import root_agent
    
    print("📦 Loading MLE-STAR agent...")
    agent = root_agent
    
    print("✅ Agent built successfully!")
    
    # Формуємо задачу для iris
    task_prompt = """I have a machine learning task for you.

Task: iris
Location: ./machine_learning_engineering/tasks/iris/

Please build a high-quality machine learning pipeline for this classification task.
The task files are already prepared in the tasks folder.

Start by reading the task description and data files, then build an optimal pipeline.
"""
    
    print("\n📤 Sending task to MLE-STAR agent...")
    print(f"Prompt:\n{task_prompt}\n")
    
    # Викликаємо агента
    print("⏳ Agent is processing (this may take 30 min - 2 hours)...\n")
    
    response = agent.run(task_prompt)
    
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    print(response)
    print("="*80)
    
    # Перевіряємо workspace
    workspace_dir = Path("./machine_learning_engineering/workspace/iris")
    if workspace_dir.exists():
        print(f"\n✅ Workspace created: {workspace_dir}")
        files = list(workspace_dir.glob("*.py"))
        print(f"   Generated files: {len(files)}")
        for f in files:
            print(f"   - {f.name}")
    else:
        print(f"\n⚠️  Workspace not found: {workspace_dir}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("RUN COMPLETE")
print("="*80)
