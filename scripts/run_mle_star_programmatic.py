#!/usr/bin/env python
"""
Програмний запуск MLE-STAR для iris датасету.
Використовує ADK API замість інтерактивного CLI.

Автор: Фефелов Ілля Олександрович
"""

import sys
import os
import argparse
from pathlib import Path

# Додаємо MLE-STAR до шляху
mle_star_root = Path(__file__).parent.parent / "adk-samples" / "python" / "agents" / "machine-learning-engineering"
sys.path.insert(0, str(mle_star_root))

# Змінюємо робочу директорію на MLE-STAR root
os.chdir(mle_star_root)

def load_env(env_file: Path):
    if not env_file.exists():
        return
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k, v)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset to generate for (overrides MLESTAR_TARGET_DATASET)')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to .env containing GEMINI_API_TOKEN or GOOGLE_API_KEY')
    parser.add_argument('--model', type=str, default=None, help='Root agent model to use (e.g. gemini-2.5-flash-lite)')
    return parser.parse_args()


# Налаштування environment
os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', '0')  # Використовуємо ML Dev backend

args = parse_args()

# Load .env file if present
load_env(Path(args.env_file))

# Map GEMINI_API_TOKEN to GOOGLE_API_KEY when appropriate
if 'GEMINI_API_TOKEN' in os.environ and 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']

# Allow specifying the root model via CLI or env
if args.model:
    os.environ['ROOT_AGENT_MODEL'] = args.model
else:
    os.environ.setdefault('ROOT_AGENT_MODEL', 'gemini-2.5-flash-lite')

print("="*80)
print("PROGRAMMATIC MLE-STAR RUN")
print("="*80)
print(f"Working directory: {os.getcwd()}")
print(f"Model: {os.environ.get('ROOT_AGENT_MODEL')}")
print(f"GOOGLE_API_KEY present: {'GOOGLE_API_KEY' in os.environ}")
print(f"GEMINI_API_TOKEN present: {'GEMINI_API_TOKEN' in os.environ}")
print("="*80)

try:
    # Імпортуємо агента
    from machine_learning_engineering.agent import root_agent
    
    print("📦 Loading MLE-STAR agent...")
    agent = root_agent
    
    print("✅ Agent built successfully!")
    
    # Figure out which dataset to run
    dataset = args.dataset or os.environ.get('MLESTAR_TARGET_DATASET') or 'iris'

    # Формуємо задачу
    task_prompt = f"""I have a machine learning task for you.

Task: {dataset}
Location: ./machine_learning_engineering/tasks/{dataset}/

Please build a high-quality machine learning pipeline for this classification task.
The task files are already prepared in the tasks folder.

Start by reading the task description and data files, then build an optimal pipeline.
"""
    
    print("\n📤 Sending task to MLE-STAR agent...")
    print(f"Prompt:\n{task_prompt}\n")
    
    # Викликаємо агента using ADK Runner API
    print("⏳ Agent is processing (this may take 30 min - 2 hours)...\n")
    try:
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        import asyncio

        runner = InMemoryRunner(agent=agent, app_name="machine-learning-engineering")

        async def run_agent():
            # create a session for this run
            session = await runner.session_service.create_session(app_name=runner.app_name, user_id="script_user")
            content = types.Content(parts=[types.Part(text=task_prompt)], role="user")
            response_text = ""
            async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=content):
                # event.content.parts is a list of parts, take first if present
                if event.content and event.content.parts and event.content.parts[0].text:
                    response_text = event.content.parts[0].text
            return response_text

        response = asyncio.run(run_agent())
        print("\n" + "="*80)
        print("AGENT RESPONSE")
        print("="*80)
        print(response)
    except Exception as e:
        print(f"❌ Error invoking ADK runner: {e}")
        import traceback
        traceback.print_exc()
        raise
    print("="*80)
    
    # Перевіряємо workspace
    workspace_dir = Path("./machine_learning_engineering/workspace") / dataset
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
