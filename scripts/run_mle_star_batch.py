#!/usr/bin/env python
"""
Запускає MLE-STAR для всіх 4 датасетів та моніторить прогрес.

ВАЖЛИВО: Цей скрипт запускає офіційний MLE-STAR Agent Development Kit,
який виконує повний цикл:
1. Initialization Agent: пошук SOTA моделей, генерація кандидатів
2. Refinement Agent: ablation + inner loop (5-10 ітерацій покращення)
3. Ensemble Agent: комбінування моделей
4. Robustness: дебагінг, перевірка витоків даних

Очікуваний час: 2-6 годин на датасет (можна паралелити).

Автор: Фефелов Ілля Олександрович
"""

import subprocess
import os
import time
from pathlib import Path
from datetime import datetime
import argparse


def run_mle_star(task_name: str, mle_star_root: Path) -> subprocess.Popen:
    """
    Запускає MLE-STAR для одного датасету у фоновому режимі.
    
    Args:
        task_name: Назва task (breast_cancer, wine, digits, iris)
        mle_star_root: Шлях до MLE-STAR директорії
        
    Returns:
        subprocess.Popen: Процес, що виконується
    """
    log_file = mle_star_root / f"logs/{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Update config.py with task_name before running
    config_path = mle_star_root / "machine_learning_engineering" / "shared_libraries" / "config.py"
    if config_path.exists():
        config_content = config_path.read_text(encoding='utf-8')
        # Replace task_name in DefaultConfig
        import re
        config_content = re.sub(
            r'task_name: str = "[^"]*"',
            f'task_name: str = "{task_name}"',
            config_content
        )
        # Reduce parallel API pressure for stability
        config_content = re.sub(
            r'num_solutions: int = \d+',
            'num_solutions: int = 1',
            config_content
        )
        config_content = re.sub(
            r'num_model_candidates: int = \d+',
            'num_model_candidates: int = 1',
            config_content
        )
        config_path.write_text(config_content, encoding='utf-8')
        print(f"   Updated config.py: task_name='{task_name}'")
    
    # Create replay JSON for non-interactive execution
    replay_json = {
        "state": {},
        "queries": [
            "Run the machine learning engineering task for the configured dataset. Generate initialization code, perform refinement, and produce final submission."
        ]
    }
    replay_file = mle_star_root / "logs" / f"{task_name}_replay.json"
    import json
    with open(replay_file, 'w') as f:
        json.dump(replay_json, f)
    
    cmd = [
        "adk", "run", "machine_learning_engineering",
        "--replay", str(replay_file)
    ]
    # Environment: disable parallel execution inside ADK agents to avoid TLS issues
    env = os.environ.copy()
    env["ADK_DISABLE_PARALLEL"] = env.get("ADK_DISABLE_PARALLEL", "1")
    
    print(f"🚀 Starting MLE-STAR for {task_name}...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log file: {log_file}")
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            cwd=mle_star_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    
    return process


def monitor_processes(processes: dict):
    """
    Моніторить запущені процеси MLE-STAR.
    
    Args:
        processes: Dict[task_name -> Popen]
    """
    print("\n" + "="*80)
    print("MONITORING MLE-STAR PROCESSES")
    print("="*80)
    
    while any(p.poll() is None for p in processes.values()):
        time.sleep(30)  # Перевірка кожні 30 секунд
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
        for task_name, process in processes.items():
            if process.poll() is None:
                print(f"  ⏳ {task_name}: Running...")
            else:
                exit_code = process.returncode
                if exit_code == 0:
                    print(f"  ✅ {task_name}: Completed successfully")
                else:
                    print(f"  ❌ {task_name}: Failed (exit code {exit_code})")
    
    print("\n" + "="*80)
    print("ALL PROCESSES FINISHED")
    print("="*80)
    
    # Підсумок
    for task_name, process in processes.items():
        workspace_dir = Path(__file__).parent.parent / "adk-samples" / "python" / "agents" / "machine-learning-engineering" / "machine_learning_engineering" / "workspace" / task_name
        
        if workspace_dir.exists():
            files = list(workspace_dir.glob("*.py"))
            print(f"\n{task_name}:")
            print(f"  Workspace: {workspace_dir}")
            print(f"  Generated files: {len(files)}")
            if files:
                for f in files:
                    print(f"    - {f.name}")
        else:
            print(f"\n{task_name}:")
            print(f"  ⚠️  Workspace not found: {workspace_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MLE-STAR agent for sklearn datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['iris', 'breast_cancer', 'wine', 'digits'],
        help='List of tasks to run (order: smallest to largest recommended)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run all tasks in parallel (faster but requires more memory)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only run iris (quick check)'
    )
    parser.add_argument(
        '--gbdt',
        action='store_true',
        help='Shortcut to run the gradient-boosted tasks (iris_gbdt, wine_gbdt, digits_gbdt)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.tasks = ['iris']
        print("🧪 TEST MODE: Running only iris dataset")
    elif args.gbdt:
        args.tasks = ['iris_gbdt', 'wine_gbdt', 'digits_gbdt']
        print("🌳 GBDT MODE: Running iris_gbdt, wine_gbdt, digits_gbdt")
    
    # Визначення шляху до MLE-STAR
    mle_star_root = Path(__file__).parent.parent / "adk-samples" / "python" / "agents" / "machine-learning-engineering"
    
    if not mle_star_root.exists():
        print(f"❌ MLE-STAR not found at: {mle_star_root}")
        print("   Make sure adk-samples is cloned in the project root")
        return
    
    print("="*80)
    print("MLE-STAR BATCH RUN")
    print("="*80)
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"MLE-STAR root: {mle_star_root}")
    print("="*80)
    
    processes = {}
    
    if args.parallel:
        # Паралельний запуск всіх task
        for task_name in args.tasks:
            processes[task_name] = run_mle_star(task_name, mle_star_root)
            time.sleep(5)  # Невелика затримка між запусками
        
        # Моніторинг всіх процесів
        monitor_processes(processes)
    
    else:
        # Послідовний запуск (один за одним)
        for task_name in args.tasks:
            print(f"\n{'='*80}")
            print(f"TASK: {task_name.upper()}")
            print(f"{'='*80}")
            
            max_attempts = 3
            attempt = 1
            last_rc = None
            while attempt <= max_attempts:
                print(f"🚀 Attempt {attempt}/{max_attempts} for {task_name}...")
                process = run_mle_star(task_name, mle_star_root)
                processes[task_name] = process
                print(f"⏳ Waiting for {task_name} to complete...")
                process.wait()
                last_rc = process.returncode
                if last_rc == 0:
                    print(f"✅ {task_name} completed successfully on attempt {attempt}!")
                    break
                else:
                    print(f"❌ {task_name} failed (exit={last_rc}) on attempt {attempt}")
                    if attempt < max_attempts:
                        print("   Retrying after 10s due to transient network error...")
                        time.sleep(10)
                attempt += 1
            if last_rc and last_rc != 0:
                print(f"❌ {task_name} failed after {max_attempts} attempts. Check logs in: {mle_star_root}/logs/")
    
    print("\n" + "="*80)
    print("BATCH RUN COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check generated pipelines in:")
    print(f"   {mle_star_root}/machine_learning_engineering/workspace/")
    print("2. Copy pipelines to our project:")
    print("   python scripts/extract_mle_star_pipelines.py")
    print("3. Run ablation on MLE-STAR pipelines:")
    print("   python scripts/main_experiment.py --skip-generation --n-runs 5")
    print("="*80)


if __name__ == "__main__":
    main()
