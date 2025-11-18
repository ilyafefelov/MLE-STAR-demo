from importlib import util
from pathlib import Path
from src.mle_star_ablation import mle_star_generated_pipeline as mgp

path = Path('generated_pipelines/pipeline_iris_flash_lite.py')
if not path.exists():
    print('File not found:', path)
    exit(1)

spec = util.spec_from_file_location('pip_mod', str(path))
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Find a builder
builder = None
for name in ('build_full_pipeline', 'create_model_pipeline', 'create_pipeline'):
    if hasattr(mod, name):
        builder = getattr(mod, name)
        break
if builder is None:
    print('No builder found in', path)
    exit(1)

try:
    pipeline = None
    try:
        pipeline = builder(random_state=42)
    except TypeError:
        pipeline = builder()
    pipeline = mgp._ensure_pipeline_object(pipeline)
    info = mgp.inspect_pipeline(pipeline)
    print('Inspect info:', info)
    print('Ablation meaningful?', mgp.is_ablation_meaningful(pipeline))
except Exception as e:
    print('Inspect failed', e)
