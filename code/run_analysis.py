"""Run the primary, contextual and prespecified sensitivity analyses."""
import argparse
import traceback

from runtime import OUTPUT, log, save_json
from data_preparation import load_data, specifications
from model_functions import fit_linear

p=argparse.ArgumentParser()
p.add_argument('--components', action='store_true', help='Also run the ordinal and categorical activity checks.')
p.add_argument('--resume', action='store_true', help='Retain completed model JSON files.')
args=p.parse_args()
_,d,s=load_data()
specs=specifications(d,s)
save_json('audit/specification_inventory.json',[{'name':v['name'],'n':len(v['data']),'clusters':v['data'].group.nunique(),'exposures':v['exposures'],'covariates':v['covariates'],'coding':v['coding']} for v in specs])
for spec in specs:
    if args.resume and (OUTPUT/'models'/(spec['name']+'.json')).exists():
        log('Already complete, retaining '+spec['name']);continue
    try:fit_linear(spec)
    except Exception as exc:
        save_json('audit/failure_'+spec['name']+'.json',{'name':spec['name'],'error':str(exc),'traceback':traceback.format_exc(),'status':'Not silently omitted; technical resolution required'})
        raise
if args.components:
    from supplementary_models import run_components
    run_components(s)
log('Requested linear specifications complete; contextual models require all six validated objective indicators.')
