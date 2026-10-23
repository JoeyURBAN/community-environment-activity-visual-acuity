"""Fixed ordinal/category checks and adjusted secondary regressions."""
from runtime import np, pd, save_json, save_csv, log
from config import SUBJECTIVE, COVARIATES
from data_preparation import categorical_replace
from model_functions import ols, joint_test, holm, diagnostics
import warnings
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.sandwich_covariance import cov_cluster


def fit_ordinal(s,outcome):
    predictors=SUBJECTIVE+COVARIATES
    means=s[predictors].mean();sd=s[predictors].std(ddof=1)
    x=(s[predictors]-means)/sd
    assert np.linalg.matrix_rank(x)==x.shape[1] and (sd>0).all()
    y=pd.Series(pd.Categorical(s[outcome].astype(int),categories=range(1,7),ordered=True),index=s.index)
    model=OrderedModel(y,x,distr='logit');attempts=[];fit=None
    for method in ['bfgs','lbfgs']:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            candidate=model.fit(method=method,maxiter=3000,disp=False,
                                **({'gtol':1e-8} if method=='bfgs' else {'pgtol':1e-8,'factr':1e4}))
        attempt={'method':method,'converged':bool(candidate.mle_retvals.get('converged',False)),
                 'log_likelihood':float(candidate.llf),'warnings':[str(w.message) for w in caught]}
        attempts.append(attempt);fit=candidate
        if attempt['converged']:break
    if not attempts[-1]['converged']:raise RuntimeError(f'{outcome} ordered logit did not converge: {attempts}')
    scores=model.score_obs(fit.params);hessian=model.hessian(fit.params);bread=np.linalg.inv(-hessian)
    groups=s.group.to_numpy();unique=np.unique(groups);g=len(unique);n=len(s);k=len(fit.params)
    grouped_scores=np.array([scores[groups==v].sum(axis=0) for v in unique])
    correction=g/(g-1)*(n-1)/(n-k)
    cov=correction*bread@(grouped_scores.T@grouped_scores)@bread
    cov=(cov+cov.T)/2
    library=cov_cluster(fit,groups,use_correction=True)
    assert np.allclose(cov,library,rtol=1e-5,atol=1e-7)
    assert (np.diag(cov)>0).all() and np.isfinite(cov).all()
    se=np.sqrt(np.diag(cov));critical=stats.t.ppf(.975,g-1);coef=np.asarray(fit.params)
    lo=coef-critical*se;hi=coef+critical*se;p=2*stats.t.sf(abs(coef/se),g-1)
    rows=[]
    for j,name in enumerate(predictors):
        rows.append({'predictor':name,'log_OR_per_SD':coef[j],'cluster_se_log_OR_per_SD':se[j],
                     'OR_per_SD':np.exp(coef[j]),'ci95_OR_per_SD':[np.exp(lo[j]),np.exp(hi[j])],
                     'log_OR_per_unit':coef[j]/sd[name],'cluster_se_log_OR_per_unit':se[j]/sd[name],
                     'OR_per_unit':np.exp(coef[j]/sd[name]),'ci95_OR_per_unit':[np.exp(lo[j]/sd[name]),np.exp(hi[j]/sd[name])],
                     'p':p[j],'t':coef[j]/se[j],'input_mean':means[name],'input_SD':sd[name]})
    thresholds=[]
    for j in range(len(predictors),k):
        thresholds.append({'parameter':fit.params.index[j],'estimate_transformed_parameter':coef[j],
                           'cluster_se_transformed_parameter':se[j],'note':'First cut point then log differences; not slope odds ratios.'})
    prob=np.asarray(model.predict(fit.params,exog=x))
    assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
    # These descriptive fits expose sparse-category / parallel-slope instability.
    # They are not additional confirmatory tests and do not choose a new model.
    binary=[]
    for threshold in range(1,6):
        yy=(s[outcome]>threshold).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                bf=sm.GLM(yy,sm.add_constant(x,has_constant='add'),family=sm.families.Binomial()).fit(maxiter=300)
                unstable=(not bf.converged) or bool(np.max(np.abs(bf.params.iloc[1:]))>10) or min(yy.sum(),len(yy)-yy.sum())<len(predictors)+1
                diag={'threshold':threshold,'n_above':int(yy.sum()),'n_at_or_below':int(len(yy)-yy.sum()),
                      'converged':bool(bf.converged),'unstable_or_sparse':unstable,
                      'slopes_log_OR_per_SD':bf.params.iloc[1:].to_dict()}
            except Exception as exc:diag={'threshold':threshold,'status':'failed_diagnostic','reason':str(exc),'unstable_or_sparse':True}
        diag['warnings']=[str(w.message) for w in caught];binary.append(diag)
    result={'name':'ordinal_'+outcome,'n':n,'clusters':g,'cluster_test_df':g-1,'link':'logit',
            'parameter_count_including_thresholds':k,'CRV1_multiplier':correction,
            'interpretation':'Odds of being above any category threshold; adjusted association, not duration in minutes.',
            'coding':'Ordered categories 1..6; no intercept; all input predictors centred and divided by sample SD.',
            'attempts':attempts,'coefficients':rows,'threshold_parameters':thresholds,
            'actual_cutpoints':model.transform_threshold_params(fit.params)[1:-1].tolist(),
            'log_likelihood':fit.llf,'aic':fit.aic,'bic':fit.bic,
            'diagnostics':{'category_counts':s[outcome].value_counts().sort_index().to_dict(),
                           'score_max_absolute':float(np.abs(scores.sum(axis=0)).max()),
                           'observed_information_min_eigenvalue':float(np.linalg.eigvalsh(-hessian).min()),
                           'cluster_covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
                           'manual_vs_library_covariance_max_difference':float(np.max(np.abs(cov-library))),
                           'probability_row_sum_max_error':float(np.max(np.abs(prob.sum(axis=1)-1))),
                           'threshold_binary_checks':binary,
                           'proportional_odds_assessment':'Threshold-specific diagnostic slopes only; sparse tails limit assessment. No claim that the parallel-odds assumption has been established.'}}
    save_csv(f'tables/ordinal_{outcome}_cluster_covariance.csv',pd.DataFrame(cov,index=fit.params.index,columns=fit.params.index).reset_index(names='parameter'))
    return result


def run_components(s):
    outputs={}
    for outcome in ['Outdoor','Screen']:
        log('Fitting ordinal '+outcome);outputs['ordinal_'+outcome]=fit_ordinal(s,outcome)
    # Supplementary family: the 26 environmental slopes across the two ordinal checks.
    rows=[r for y in ['Outdoor','Screen'] for r in outputs['ordinal_'+y]['coefficients'] if r['predictor'] in SUBJECTIVE]
    holm(rows,'p_holm_ordinal_environment_family')
    for y in ['Outdoor','Screen']:
        outputs['ordinal_'+y]['multiplicity']={'environmental_slopes_across_both_outcomes':len(rows),'status':'supplementary descriptive sensitivity family, separate from linear primary families'}
        save_json(f"components/ordinal_{y}.json",outputs['ordinal_'+y])
    cat,ot,ometa=categorical_replace(s,'Outdoor','OutdoorCategory',1)
    cat,st,smeta=categorical_replace(cat,'Screen','ScreenCategory',1)
    rec,fit,robust=ols(cat,'VA',SUBJECTIVE+COVARIATES+ot+st)
    joint=[{'activity':'Outdoor',**joint_test(robust,ot)},{'activity':'Screen',**joint_test(robust,st)}]
    holm(joint,'p_holm_two_activity_category_tests')
    rec.update({'name':'VA_activity_categories','coding':[ometa,smeta],'joint_tests':joint,
                'diagnostics':diagnostics(fit),'scope':'Category contribution to VA equation only; not combined with ordinal activity equations for indirect effects.'})
    outputs['VA_activity_categories']=rec;save_json('components/VA_activity_categories.json',rec)
    return outputs
