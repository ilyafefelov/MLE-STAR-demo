#!/usr/bin/env python
"""
Monitor MLE-STAR ADK workspace for a generated pipeline and automatically
copy to `model_comparison_results`, validate, and run ablation once ready.

Usage:
  python scripts/monitor_generated_workspace_and_run_ablation.py --dataset iris --timeout 3600 --n-runs 3 --deterministic
"""
import argparse
import time
from pathlib import Path
import shutil
import subprocess
import os
import sys
import json
import traceback
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=3600, help='Max seconds to wait')
    parser.add_argument('--interval', type=int, default=30, help='Poll interval seconds')
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--env-file', type=str, default='.env')
    parser.add_argument('--lock-dir', type=str, default=None, help='Lock directory (defaults to monitor_logs/<dataset>.lock)')
    parser.add_argument('--lock-stale-seconds', type=int, default=600, help='Seconds after which a lock is considered stale')
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression'], default=None, help='Force task-type when running ablation')
    parser.add_argument('--wait-for-lock', action='store_true', help='If lock exists, wait until released instead of exiting')
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    repo_root = Path(__file__).parent.parent
    workspace_base = repo_root / 'adk-samples' / 'python' / 'agents' / 'machine-learning-engineering' / 'machine_learning_engineering' / 'workspace'
    workspace_dir = workspace_base / ds
    timeout_at = time.time() + args.timeout
    # monitor logs dir
    monitor_dir = repo_root / 'monitor_logs'
    monitor_dir.mkdir(parents=True, exist_ok=True)

    progress_log_path = monitor_dir / f'{ds}_progress.log'

    def append_progress_log(line: str):
        try:
            with open(progress_log_path, 'a', encoding='utf-8') as lf:
                lf.write(line + '\n')
        except Exception as e:
            print('Could not append to progress log:', e)

    def write_monitor_state(state: str, details: dict | None = None):
        payload = {
            'dataset': ds,
            'state': state,
            'timestamp': datetime.now().isoformat(),
        }
        if details:
            payload.update(details)
        try:
            result_json_path = monitor_dir / f'{ds}_last_ablation.json'
            result_json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        except Exception as e:
            print('Could not write monitor state file:', e)
        # Also append a short text log entry for easy CI visibility
        try:
            details_short = ''
            if details:
                # keep it compact for log
                details_short = ' ' + ' '.join([f'{k}={v}' for k, v in details.items()])
            append_progress_log(f"{datetime.now().isoformat()} | {state.upper()} |{details_short}")
        except Exception as e:
            print('Could not update plaintext progress log:', e)

    print(f'Waiting for ADK workspace pipeline in: {workspace_dir}')
    processed_files = {}  # mapping from file path -> {'mtime': mtime, 'hash': sha256, 'last_processed_ts': float}
    file_cooldown_seconds = 60  # do not reprocess same file within this cooldown
    # Lock handling
    if args.lock_dir:
        lock_dir = Path(args.lock_dir)
    else:
        lock_dir = monitor_dir / f'{ds}.lock'
    lock_dir = lock_dir.resolve()
    lock_meta = lock_dir / 'lock.json'
    lock_stale_seconds = args.lock_stale_seconds

    def is_process_alive(pid: int) -> bool:
        try:
            if os.name == 'nt':
                # Windows: use tasklist to check for PID
                out = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'], stderr=subprocess.DEVNULL)
                return str(pid) in out.decode('utf-8', errors='ignore')
            else:
                os.kill(pid, 0)
                return True
        except Exception:
            return False

    def acquire_lock():
        # attempt to create lock dir atomically
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            # Write metadata
            meta = {'pid': os.getpid(), 'timestamp': datetime.now().isoformat()}
            try:
                lock_meta.write_text(json.dumps(meta), encoding='utf-8')
            except Exception:
                pass
            append_progress_log(f"LOCK_ACQUIRED pid={os.getpid()} lock_dir={lock_dir}")
            return True
        except FileExistsError:
            # lock exists; check for staleness
            try:
                text = lock_meta.read_text(encoding='utf-8')
                data = json.loads(text)
                pid = int(data.get('pid', 0))
                ts = data.get('timestamp')
                stale = False
                if ts:
                    try:
                        t = datetime.fromisoformat(ts)
                        if (datetime.now() - t).total_seconds() > lock_stale_seconds:
                            stale = True
                    except Exception:
                        stale = False
                # if owner process is dead OR stale timestamp, remove lock and try again
                if not is_process_alive(pid) or stale:
                    append_progress_log(f"LOCK_STALE_OR_ORPHAN pid={pid} stale={stale}")
                    try:
                        # remove lock
                        for p in lock_dir.glob('*'):
                            p.unlink()
                        lock_dir.rmdir()
                    except Exception:
                        pass
                    # try to create again
                    try:
                        lock_dir.mkdir(parents=True, exist_ok=False)
                        meta = {'pid': os.getpid(), 'timestamp': datetime.now().isoformat()}
                        try:
                            lock_meta.write_text(json.dumps(meta), encoding='utf-8')
                        except Exception:
                            pass
                        append_progress_log(f"LOCK_ACQUIRED_AFTER_STALE pid={os.getpid()} lock_dir={lock_dir}")
                        return True
                    except Exception:
                        return False
                # Lock not acquirable
                append_progress_log(f"LOCK_HELD_BY pid={pid} lock_dir={lock_dir}")
                return False
            except Exception:
                # can't read metadata; treat as held
                append_progress_log(f"LOCK_HELD_UNKNOWN lock_dir={lock_dir}")
                return False

    def release_lock():
        try:
            if lock_meta.exists():
                lock_meta.unlink()
            if lock_dir.exists():
                # remove contents and rmdir
                for p in lock_dir.glob('*'):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    lock_dir.rmdir()
                except Exception:
                    # try to force remove any inner directories recursively
                    import shutil
                    try:
                        shutil.rmtree(lock_dir)
                    except Exception:
                        pass
            append_progress_log(f"LOCK_RELEASED pid={os.getpid()} lock_dir={lock_dir}")
        except Exception:
            pass
    try:
        write_monitor_state('started', {'workspace_dir': str(workspace_dir)})
        # Acquire lock to avoid parallel monitors
        acquired = acquire_lock()
        if not acquired:
            if args.wait_for_lock:
                append_progress_log(f"WAITING_FOR_LOCK pid={os.getpid()} lock_dir={lock_dir}")
                # Loop waiting until lock released or timeout
                while not acquire_lock():
                    if time.time() >= timeout_at:
                        print(f"Timeout while waiting for lock at {lock_dir}. Exiting.")
                        write_monitor_state('error', {'error': f'timeout_waiting_for_lock {str(lock_dir)}'})
                        sys.exit(3)
                    time.sleep(args.interval)
                append_progress_log(f"LOCK_ACQUIRED_AFTER_WAIT pid={os.getpid()} lock_dir={lock_dir}")
            else:
                print(f"Could not acquire lock at {lock_dir}; another monitor may be running. Exiting.")
                write_monitor_state('error', {'error': f'could_not_acquire_lock {str(lock_dir)}'})
                sys.exit(3)
        # Try to load processed file state to avoid reprocessing after monitor restarts
        processed_file_state_path = monitor_dir / f'{ds}_processed_files.json'
        if processed_file_state_path.exists():
            try:
                loaded = json.loads(processed_file_state_path.read_text(encoding='utf-8'))
                # no need for converting; assume structure matches
                processed_files.update(loaded)
            except Exception:
                # ignore errors and start fresh
                pass
        while time.time() < timeout_at:
            # Sometimes agent names the folder with task variations; look for directories that contain dataset as prefix
            candidate_dirs = [p for p in workspace_base.iterdir() if p.is_dir() and p.name.startswith(ds)]
            py_files = []
            for candidate in candidate_dirs:
                # Search recursively for pipeline files, agent may write them into nested subfolders
                py_files.extend(list(candidate.rglob('*.py')))
            if py_files:
                # use largest file heuristic
                best = max(py_files, key=lambda p: p.stat().st_size)
                print(f'Found candidate generated file: {best}')
                write_monitor_state('detected', {'src': str(best)})
                # compute mtime and short hash to avoid reprocessing the same file repeatedly
                try:
                    cur_mtime = best.stat().st_mtime
                except Exception:
                    cur_mtime = None

                def file_hash(path: Path) -> str:
                    import hashlib
                    h = hashlib.sha256()
                    try:
                        with open(path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b''):
                                h.update(chunk)
                        return h.hexdigest()[:12]
                    except Exception:
                        return ''

                cur_hash = file_hash(best)
                out_file = repo_root / 'model_comparison_results' / f'gemini_live_{ds}.py'
                out_file.parent.mkdir(parents=True, exist_ok=True)
                previous = processed_files.get(str(best))
                if previous:
                    prev_time = previous.get('mtime')
                    prev_hash = previous.get('hash')
                    last_processed_ts = previous.get('last_processed_ts')
                    # If mtime and hash didn't change and we're within cooldown, skip
                    if cur_mtime == prev_time and cur_hash == prev_hash:
                        if last_processed_ts and (time.time() - last_processed_ts) < file_cooldown_seconds:
                            # last_processed_ts is within cooldown; skip repeated processing to avoid spam
                            print(f'Existing destination {out_file} is up to date (skipping)')
                            # don't busy-loop; sleep interval then continue
                            time.sleep(args.interval)
                            continue
                out_file = repo_root / 'model_comparison_results' / f'gemini_live_{ds}.py'
                out_file.parent.mkdir(parents=True, exist_ok=True)
                # Only copy if out_file doesn't exist or source is newer
                if out_file.exists():
                    if out_file.stat().st_mtime >= best.stat().st_mtime:
                        # record the processed file and mtime (so we don't spam logs)
                        processed_files[str(best)] = {'mtime': cur_mtime, 'hash': cur_hash, 'last_processed_ts': time.time()}
                        # Try to persist processed state
                        try:
                            processed_file_state_path.write_text(json.dumps(processed_files, indent=2), encoding='utf-8')
                        except Exception:
                            pass
                        print(f'Existing destination {out_file} is up to date (skipping)')
                        time.sleep(args.interval)
                        continue
                    else:
                        print(f'Destination {out_file} is older than {best}, overwriting')
                shutil.copy(best, out_file)
                print(f'Copied {best.name} to {out_file}')
                # Create an in-repo wrapper that safely imports the generated module and exposes build_full_pipeline
                wrapper_path = out_file.parent / f'gemini_live_{ds}_wrapped.py'
                wrapper_text = f"""
    import importlib.util
    import sys
    from pathlib import Path

    # Load the generated module by path without running fit() automatically
    spec = importlib.util.spec_from_file_location('gemini_live_{ds}', r'{out_file}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['gemini_live_{ds}'] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # If the module executed training code during import, we will not try to import further here
        pass

    builder = getattr(mod, 'build_full_pipeline', None) or getattr(mod, 'create_model_pipeline', None) or getattr(mod, 'create_pipeline', None)

    def build_full_pipeline(*args, **kwargs):
        if builder is None:
            raise RuntimeError('Could not find builder function in generated module')
        p = builder(*args, **kwargs)
        if isinstance(p, tuple):
            for el in p:
                if hasattr(el, 'fit'):
                    return el
            return p[0]
        return p
    """
                try:
                    wrapper_path.write_text(wrapper_text, encoding='utf-8')
                    print(f'Created safe wrapper at {wrapper_path}')
                    write_monitor_state('copy_done', {'dest': str(out_file), 'wrapper': str(wrapper_path)})
                    # update processed_files to reflect that we've handled this
                    processed_files[str(best)] = {'mtime': cur_mtime, 'hash': cur_hash, 'last_processed_ts': time.time()}
                    try:
                        processed_file_state_path.write_text(json.dumps(processed_files, indent=2), encoding='utf-8')
                    except Exception:
                        pass
                except Exception as e:
                    print(f'Could not write wrapper: {e}')
                    write_monitor_state('error', {'error': str(e)})
                # Inspect
                subprocess.run([sys.executable, str(repo_root / 'scripts' / 'inspect_generated_pipelines.py'), '--dir', str(candidate)])
                # Validate
                # Validate the copy or the wrapper if created
                pipeline_to_validate = str(wrapper_path) if 'wrapper_path' in locals() and wrapper_path.exists() else str(out_file)
                # Validate initial wrapper or out file
                write_monitor_state('validating', {'pipeline_file': pipeline_to_validate})
                res = subprocess.run([sys.executable, str(repo_root / 'scripts' / 'validate_generated_pipeline.py'), '--pipeline-file', pipeline_to_validate, '--cv', '2', '--random-state', '42'])
                # If validation failed due to training code executed at import, create an AST-based wrapper and fallback
                # The validator returns non-zero on failure; we will re-run and catch failures by reading exit code
                ast_wrapper = None
                auto_wrapper = None
                if res.returncode != 0:
                        # Try AST-based extraction wrapper
                        extract_cmd = [sys.executable, str(repo_root / 'scripts' / 'extract_pipeline_wrapper.py'), '--src', str(out_file), '--out', str(out_file.parent / f'gemini_live_{ds}_wrapper_auto.py')]
                        print('Attempting AST-based wrapper extraction:', extract_cmd)
                        subprocess.run(extract_cmd, check=False)
                        ast_wrapper = out_file.parent / f'gemini_live_{ds}_wrapper_auto.py'
                        if ast_wrapper and ast_wrapper.exists():
                            # Validate wrapper
                            print('Validating AST-extracted wrapper:', ast_wrapper)
                            write_monitor_state('ast_extracted', {'ast_wrapper': str(ast_wrapper)})
                            subprocess.run([sys.executable, str(repo_root / 'scripts' / 'validate_generated_pipeline.py'), '--pipeline-file', str(ast_wrapper), '--cv', '2', '--random-state', '42'])
                            pipeline_to_validate = str(ast_wrapper)
                        else:
                            # Fallback: auto builder with minimal pipeline
                            auto_cmd = [sys.executable, str(repo_root / 'scripts' / 'auto_make_pipeline_wrapper.py'), '--src', str(out_file), '--out', str(out_file.parent / f'gemini_live_{ds}_pipeline_wrapper.py')]
                            print('Creating fallback pipeline wrapper:', auto_cmd)
                            subprocess.run(auto_cmd, check=False)
                            auto_wrapper = out_file.parent / f'gemini_live_{ds}_pipeline_wrapper.py'
                        if auto_wrapper and auto_wrapper.exists():
                            subprocess.run([sys.executable, str(repo_root / 'scripts' / 'validate_generated_pipeline.py'), '--pipeline-file', str(auto_wrapper), '--cv', '2', '--random-state', '42'])
                            pipeline_to_validate = str(auto_wrapper)
                            write_monitor_state('auto_wrapped', {'auto_wrapper': str(auto_wrapper)})
                # Run ablation
                cmd = [sys.executable, str(repo_root / 'scripts' / 'run_ablation.py'), '--dataset', ds, '--n-runs', str(args.n_runs), '--pipeline-file', pipeline_to_validate]
                if args.deterministic:
                    cmd.extend(['--deterministic', '--seed', '42'])
                cmd.append('--no-plots')
                cmd.append('--verbose')
                if args.task_type:
                    cmd.extend(['--task-type', args.task_type])
                print('Running ablation:', ' '.join(cmd))
                write_monitor_state('ablation_started', {'pipeline_file': pipeline_to_validate, 'cmd': ' '.join(cmd)})
                subprocess.run(cmd)
                print('Ablation flow completed for dataset:', ds)
                write_monitor_state('ablation_completed', {'pipeline_file': pipeline_to_validate})
                # Write a monitor state file so other processes/CI/agents can pick up status
                monitor_dir = repo_root / 'monitor_logs'
                monitor_dir.mkdir(parents=True, exist_ok=True)
                result_json_path = monitor_dir / f'{ds}_last_ablation.json'
                payload = {
                    'dataset': ds,
                    'pipeline_file': pipeline_to_validate,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                }
                try:
                    result_json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                    print(f'Wrote monitor completion state to {result_json_path}')
                except Exception as e:
                    print('Could not write monitor state file:', e)
                # Release lock and exit cleanly
                try:
                    release_lock()
                except Exception:
                    pass
                append_progress_log('ABORT | release_lock called before return')
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Monitor interrupted by user.')
        write_monitor_state('interrupted', {'reason': 'KeyboardInterrupt'})
        try:
            release_lock()
        except Exception:
            pass
        sys.exit(2)
    except Exception as ex:
        print('Monitor failed with error:', ex)
        write_monitor_state('error', {'error': str(ex), 'trace': traceback.format_exc()})
        try:
            release_lock()
        except Exception:
            pass
        sys.exit(2)
    print('Timeout reached, no generated pipeline found in workspace')
    try:
        release_lock()
    except Exception:
        pass
    sys.exit(2)


if __name__ == '__main__':
    main()
