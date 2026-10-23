import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from config import BASE, OUTPUT

sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

for key, part in [('TMP', 'temp'), ('TEMP', 'temp'), ('JOBLIB_TEMP_FOLDER', 'joblib')]:
    folder = BASE / 'cache' / part
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)


def guard_write(event, args):
    """Prevent analysis scripts from writing outside this package."""
    def check(path):
        if isinstance(path, int) or path is None:
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved == Path(os.devnull).resolve():
            return
        if not resolved.is_relative_to(BASE):
            raise PermissionError(f'Write outside analysis package blocked: {resolved}')

    if event == 'open':
        path, mode, flags = args
        writable = isinstance(mode, str) and any(char in mode for char in 'wax+')
        writable = writable or (isinstance(flags, int) and flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writable:
            check(path)
    elif event in ['os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime']:
        check(args[0])
    elif event in ['os.rename', 'os.replace']:
        check(args[0])
        check(args[1])


sys.addaudithook(guard_write)

import numpy as np
import pandas as pd


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,pd.DataFrame):return clean(v.to_dict('records'))
    if isinstance(v,pd.Series):return clean(v.to_dict())
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,(int,np.integer)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,Path):return str(v)
    return v
def save_json(relative,v):
    p=OUTPUT/relative;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save_csv(relative,v):
    p=OUTPUT/relative;p.parent.mkdir(parents=True,exist_ok=True)
    (v if isinstance(v,pd.DataFrame) else pd.DataFrame(v)).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.15g')
def now():return datetime.now(timezone.utc).isoformat()
def log(message):
    line = now() + ' ' + message
    print(line, flush=True)
    path = BASE / 'logs' / 'run.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
