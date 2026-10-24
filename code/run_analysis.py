"""Run the primary, five-indicator contextual and retained sensitivity models."""
import argparse
import json
import traceback

from runtime import OUTPUT, log, save_json
from data_preparation import load_data, specifications

def compatible_result(saved, spec):
    """Reject old six-indicator results under the R2 contextual output name.

    This checks the specification, not input-file identity. Resume requires
    unchanged input data and completed results from the same analysis run.
    """
    return (
        saved.get('name') == spec['name']
        and saved.get('n') == len(spec['data'])
        and saved.get('clusters') == spec['data'].group.nunique()
        and saved.get('exposures') == spec['exposures']
        and saved.get('covariates') == spec['covariates']
        and saved.get('coding', {}) == spec['coding']
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components', action='store_true',
                        help='Also run the ordinal and categorical activity checks.')
    parser.add_argument('--greenness-comparisons', action='store_true',
                        help='Also run the original six-indicator and NDVI-substitution contextual models.')
    parser.add_argument('--resume', action='store_true',
                        help='Retain completed model JSON files with matching specifications and unchanged input data.')
    args = parser.parse_args()
    from model_functions import fit_linear
    _, data, primary = load_data()
    specs = specifications(data, primary, args.greenness_comparisons)
    save_json('audit/specification_inventory.json', [
        {'name': spec['name'], 'n': len(spec['data']),
         'clusters': spec['data'].group.nunique(), 'exposures': spec['exposures'],
         'covariates': spec['covariates'], 'coding': spec['coding']}
        for spec in specs
    ])
    for spec in specs:
        result_path = OUTPUT / 'models' / (spec['name'] + '.json')
        if args.resume and result_path.exists():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if not compatible_result(saved, spec):
                raise ValueError(
                    f"Stored result for {spec['name']} uses a different specification. "
                    'Preserve the old results and use a fresh working copy for R2.'
                )
            log('Already complete, retaining ' + spec['name'])
            continue
        try:
            fit_linear(spec)
        except Exception as exc:
            save_json('audit/failure_' + spec['name'] + '.json', {
                'name': spec['name'], 'error': str(exc),
                'traceback': traceback.format_exc(),
                'status': 'Not silently omitted; technical resolution required',
            })
            raise
    if args.components:
        from supplementary_models import run_components
        run_components(primary)
    log('Requested analyses complete. The main contextual model uses five mapped indicators; '
        'six-indicator and NDVI-substitution models are supplementary comparisons.')


if __name__ == '__main__':
    main()
