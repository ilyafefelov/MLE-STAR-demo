#!/usr/bin/env python
"""
Run preflight checks to validate environment for ADK/GenAI/agent runs.

Checks performed:
 - Loads .env (if provided) and verifies `GEMINI_API_TOKEN` or `GOOGLE_API_KEY` is set
 - Prints Python version and important installed package versions (google-genai, google-adk, vertexai)
 - Attempts to list or validate existence of asked models (e.g., gemini-2.5-flash-lite, gemini-2.1) using google.genai if credentials allow
 - Verifies presence of ADK workspace path and checks for write permissions

Usage:
  python scripts/adk_preflight.py --env-file .env --models gemini-2.5-flash-lite,gemini-2.1

"""
import argparse
import os
import sys
from pathlib import Path
import importlib
import json


def load_env(env_path: Path):
    if not env_path.exists():
        return
    with env_path.open('r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k, v)


def print_pkg_version(name: str):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, '__version__', None)
        print(f'{name} version: {version}')
    except Exception:
        print(f'{name} not installed or import failed')


def mask_value(val: str) -> str:
    if not val:
        return ''
    s = str(val)
    if len(s) <= 8:
        return f'{s[:2]}****{s[-2:]}'
    return f'{s[:4]}...{s[-4:]}'


def model_availability_check(model_name: str, verbose: bool = False) -> tuple[bool, str]:
    """Try to call google.genai `models.get` or `models.list` to verify the model is accessible.
    Returns (available, message).
    """
    try:
        genai = importlib.import_module('google.genai')
    except Exception as e:
        return False, f'google.genai not installed: {e}'
    # try to use models.get if available; else attempt to create a client
    try:
        models = getattr(genai, 'models', None)
        if models is None:
            if verbose:
                print('genai.models not present on the genai module; attributes:', [a for a in dir(genai) if not a.startswith('_')])
            return False, 'google.genai.models not present in SDK'
        # attempt using get() if available
        get_fn = getattr(models, 'get', None)
        if callable(get_fn):
            try:
                _ = get_fn(model=model_name)
                return True, f'Model {model_name} is available (models.get)'
            except Exception as e:
                msg = str(e)
                if verbose:
                    import traceback
                    traceback.print_exc()
                    print('models.get() error for {0} : {1!r}'.format(model_name, e))
                # continue to try list or generate
        # try list() if available
        list_fn = getattr(models, 'list', None)
        if callable(list_fn):
            try:
                items = list_fn()
                for m in items:
                    # models from list may be objects with name attribute
                    nm = getattr(m, 'name', str(m))
                    if nm == model_name or nm.endswith(f'/{model_name}'):
                        return True, f'Model {model_name} is available (models.list)'
            except Exception:
                if verbose:
                    import traceback
                    traceback.print_exc()
                pass
        # final fallback: attempt a tiny generate_text call with the model
        try:
            # high-level convenience API
            if hasattr(genai, 'generate_text') and callable(getattr(genai, 'generate_text')):
                try:
                    _ = genai.generate_text(model=model_name, input='hi')
                    return True, f'Model {model_name} responded to a small genai.generate_text call'
                except Exception as e:
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    print(f'genai.generate_text() returned error for {model_name}:', repr(e))
            # models.generate_text (some SDK versions expose generate under genai.models)
            models_mod = getattr(genai, 'models', None)
            if models_mod is not None and hasattr(models_mod, 'generate_text') and callable(getattr(models_mod, 'generate_text')):
                try:
                    _ = models_mod.generate_text(model=model_name, input='hi')
                    return True, f'Model {model_name} responded to a small models.generate_text call'
                except Exception as e:
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    print(f'genai.models.generate_text() returned error for {model_name}:', repr(e))
            # Some sdk shapes expose an alternative or class-based 'Models' API
            # Try to detect 'getv' at module level (older or internal name) or a 'Models' class
            getv_fn = getattr(models, 'getv', None)
            if callable(getv_fn):
                # Try both keyword and positional call styles for getv
                try:
                    _ = getv_fn(model=model_name)
                    return True, f'Model {model_name} is available (models.getv - kwargs)'
                except TypeError:
                    # try as positional arg
                    try:
                        _ = getv_fn(model_name)
                        return True, f'Model {model_name} is available (models.getv - positional)'
                    except Exception as e:
                        if verbose:
                            import traceback
                            traceback.print_exc()
                        print(f'models.getv() returned error for {model_name}:', repr(e))
                except Exception as e:
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    print(f'models.getv() returned error for {model_name}:', repr(e))
            # Try class-based Models (e.g., models.Models())
            try:
                ModelsClass = getattr(models, 'Models', None)
                if ModelsClass and callable(ModelsClass):
                    try:
                        models_client = ModelsClass()
                        # candidate names for get/list methods
                        candidates = ['get', 'getv', 'list', 'listv']
                        for c in candidates:
                            method = getattr(models_client, c, None)
                            if callable(method):
                                try:
                                    # Try model= param first, fallback to name=
                                    try:
                                        _ = method(model=model_name)
                                    except TypeError:
                                        _ = method(name=model_name)
                                    return True, f'Model {model_name} is available (models.Models().{c})'
                                except Exception as e:
                                    if verbose:
                                        import traceback
                                        traceback.print_exc()
                                    print(f'models.Models().{c}() returned error for {model_name}:', repr(e))
                    except Exception:
                        if verbose:
                            import traceback
                            traceback.print_exc()
            except Exception:
                # ignore failures in introspecting client - we'll fallback to other checks
                if verbose:
                    import traceback
                    traceback.print_exc()
            # older or alternative SDK entrypoints: Test any 'GenerationModel' or 'TextGenerationModel' constructs
            # Some SDKs offer a synchronous client-like interface that must be instantiated - skip for now to avoid side-effects
        except Exception as e2:
            if verbose:
                import traceback
                traceback.print_exc()
            return False, f'Model small generate check failed: {e2.__class__.__name__}: {e2}'
        # any other errors in the model-check block bubble up here
        try:
            # Try multiple client constructors exposed by the SDK
            client_constructors = []
            if hasattr(genai, 'Client'):
                client_constructors.append(getattr(genai, 'Client'))
            if hasattr(genai, 'client') and hasattr(genai.client, 'Client'):
                client_constructors.append(getattr(genai.client, 'Client'))

            for ClientCtor in client_constructors:
                if not callable(ClientCtor):
                    continue
                try:
                    client = ClientCtor()
                except Exception as e:
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    # Could not instantiate this client - try next
                    continue
                # Print some diagnostics for the client
                if verbose:
                    try:
                        print('\n[Client] attrs:', [n for n in dir(client) if not n.startswith('_')][:80])
                    except Exception:
                        pass
                # Try simple generate-like calls on client
                try:
                    if hasattr(client, 'generate_text') and callable(getattr(client, 'generate_text')):
                        try:
                            _ = client.generate_text(model=model_name, input='hi')
                            return True, f'Model {model_name} responded via client.generate_text'
                        except Exception:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                    if hasattr(client, 'generate') and callable(getattr(client, 'generate')):
                        try:
                            _ = client.generate(model=model_name, input='hi')
                            return True, f'Model {model_name} responded via client.generate'
                        except Exception:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                    # try chat shape
                    chat = getattr(client, 'chat', None)
                    if chat is not None and hasattr(chat, 'completions') and hasattr(chat.completions, 'create'):
                        try:
                            _ = chat.completions.create(model=model_name, messages=[{'role': 'user', 'content': 'hi'}])
                            return True, f'Model {model_name} responded via client.chat.completions.create'
                        except Exception:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                except Exception:
                    if verbose:
                        import traceback
                        traceback.print_exc()

                # Try using ``get_model`` and ``list_models`` helpers
                try:
                    models_obj = getattr(client, 'models', None)
                    if models_obj is not None:
                        # try models_obj.get / getv / list
                        try:
                            if hasattr(models_obj, 'get') and callable(getattr(models_obj, 'get')):
                                try:
                                    _ = models_obj.get(model=model_name)
                                    return True, f'Model {model_name} is available (client.models.get)'
                                except Exception:
                                    if verbose:
                                        import traceback
                                        traceback.print_exc()
                            getv_fn = getattr(models_obj, 'getv', None)
                            if callable(getv_fn):
                                # try positional
                                try:
                                    _ = getv_fn(model_name)
                                    return True, f'Model {model_name} is available (client.models.getv)'
                                except Exception:
                                    if verbose:
                                        import traceback
                                        traceback.print_exc()
                            if hasattr(models_obj, 'list') and callable(getattr(models_obj, 'list')):
                                try:
                                    items = models_obj.list()
                                    for m in items:
                                        nm = getattr(m, 'name', str(m))
                                        if nm == model_name or nm.endswith(f'/{model_name}'):
                                            return True, f'Model {model_name} is available (client.models.list)'
                                except Exception:
                                    if verbose:
                                        import traceback
                                        traceback.print_exc()
                        except Exception:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                    if hasattr(client, 'get_model') and callable(getattr(client, 'get_model')):
                        import inspect
                        sig = inspect.signature(client.get_model)
                        kwargs = {}
                        if 'model' in sig.parameters:
                            kwargs['model'] = model_name
                        elif 'name' in sig.parameters:
                            kwargs['name'] = model_name
                        try:
                            if kwargs:
                                _ = client.get_model(**kwargs)
                            else:
                                _ = client.get_model(model_name)
                            return True, f'Model {model_name} is available (client.get_model)'
                        except Exception as e:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                            print(f'client.get_model() returned error for {model_name}:', repr(e))
                    if hasattr(client, 'list_models') and callable(getattr(client, 'list_models')):
                        try:
                            items = client.list_models()
                            for m in items:
                                nm = getattr(m, 'name', str(m))
                                if nm == model_name or nm.endswith(f'/{model_name}'):
                                    return True, f'Model {model_name} is available (client.list_models)'
                        except Exception:
                            if verbose:
                                import traceback
                                traceback.print_exc()
                except Exception:
                    if verbose:
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
            return False, f'Error checking client-based model availability: {e}'
        return False, f'Model {model_name} not found by get/list or generate'
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return False, f'Error when attempting to check model: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=str, default='.env')
    parser.add_argument('--models', type=str, default='gemini-2.5-flash-lite',
                        help='Comma-separated list of model ids to check (e.g. gemini-2.5-flash-lite). 2.1 is deprecated')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose SDK and env dumps for debugging')
    parser.add_argument('--include-deprecated', action='store_true', help='Include deprecated models like gemini-2.1 in availability checks (default: skip)')
    args = parser.parse_args()

    env_path = Path(args.env_file)
    load_env(env_path)
    print('Loaded env file:', env_path if env_path.exists() else 'none')
    verbose = bool(args.verbose)
    # Map GEMINI_API_TOKEN to GOOGLE_API_KEY if needed (do this before dumping env info)
    if 'GEMINI_API_TOKEN' in os.environ and 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']
    if verbose:
        print('\nVerbose mode enabled: dumping important environment variables (masked)')
        print('GOOGLE_API_KEY:', mask_value(os.environ.get('GOOGLE_API_KEY', '')))
        print('GEMINI_API_TOKEN:', mask_value(os.environ.get('GEMINI_API_TOKEN', '')))
        print('GCLOUD_PROJECT:', os.environ.get('GCLOUD_PROJECT',''))
        print('GOOGLE_CLOUD_PROJECT:', os.environ.get('GOOGLE_CLOUD_PROJECT',''))
        print('PATH:', os.environ.get('PATH','')[:400], '...')

    # Map GEMINI_API_TOKEN to GOOGLE_API_KEY if needed
    if 'GEMINI_API_TOKEN' in os.environ and 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']
    apikey = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_TOKEN')
    print('GOOGLE_API_KEY present:' , bool(os.environ.get('GOOGLE_API_KEY')))
    print('GEMINI_API_TOKEN present:' , bool(os.environ.get('GEMINI_API_TOKEN')))
    if not apikey:
        print('ERROR: Neither GOOGLE_API_KEY nor GEMINI_API_TOKEN found in environment.')
        print('Please add your API key to the .env or env and retry.')
        sys.exit(2)

    # Print python and package versions
    print('Python version: ', sys.version)
    print_pkg_version('google.genai')
    print_pkg_version('google.adk')
    print_pkg_version('vertexai')
    print_pkg_version('google-cloud-core')

    # If verbose, show some SDK attributes that may be useful for debugging
    if verbose:
        try:
            genai = importlib.import_module('google.genai')
            print('\n[genai module]:', [n for n in dir(genai) if not n.startswith('_')][:50])
            models_attr = getattr(genai, 'models', None)
            print('[genai.models present?]:', models_attr is not None)
            if models_attr is not None:
                print('[genai.models attrs]:', [n for n in dir(models_attr) if not n.startswith('_')])
        except Exception:
            print('Could not import google.genai to list attributes (not installed or import failed)')

    # Check ADK-samples workspace exists
    repo_root = Path(__file__).parent.parent
    workspace_base = repo_root / 'adk-samples' / 'python' / 'agents' / 'machine-learning-engineering' / 'machine_learning_engineering' / 'workspace'
    print('ADK workspace base:', workspace_base)
    if not workspace_base.exists():
        print('Workspace base not found; agent may not be installed or ADK-samples not present')
    else:
        print('Workspace base exists; permissions OK? Writable:', os.access(workspace_base, os.W_OK))

    # Model checks
    for model in [m.strip() for m in args.models.split(',')]:
        if not model:
            continue
        # Respect policy: skip gemini-2.1 checks by default unless include_deprecated is set
        if model.strip() == 'gemini-2.1' and not args.include_deprecated:
            print('Skipping gemini-2.1 check (deprecated). Use --include-deprecated to force-check it.')
            continue
        ok, msg = model_availability_check(model, verbose=verbose)
        print(f'Model check for {model}: {"OK" if ok else "NOT OK"} - {msg}')

    print('\nPreflight checks complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
