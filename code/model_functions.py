"""Observed-variable parallel paths; all inference rules fixed before fitting."""
from runtime import np, save_json, log
from config import OUTPUT, N_BOOT, SEED
import warnings
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import semopy


def holm(rows,label='p_holm',source='p'):
    assert rows and all(np.isfinite(r[source]) for r in rows)
    for row,p in zip(rows,multipletests([r[source] for r in rows],method='holm')[1]):
        row[label]=float(p)


def ols(data,outcome,predictors):
    x=sm.add_constant(data[predictors].astype(float),has_constant='add')
    assert data[[outcome]+predictors+['group']].notna().all().all()
    assert np.linalg.matrix_rank(x)==x.shape[1], 'Rank-deficient analysis design'
    fit=sm.OLS(data[outcome],x).fit()
    robust=fit.get_robustcov_results(cov_type='cluster',groups=data.group,use_correction=True,df_correction=True,use_t=True)
    hc3=fit.get_robustcov_results(cov_type='HC3',use_t=True)
    rows=[];ci=robust.conf_int();ci3=hc3.conf_int();sy=data[outcome].std(ddof=1)
    for j,name in enumerate(x.columns):
        scale=data[name].std(ddof=1)/sy if name!='const' else np.nan
        rows.append({'predictor':name,'B':fit.params.iloc[j],'beta':fit.params.iloc[j]*scale,
                     'cluster_se':robust.bse[j],'cluster_se_beta':robust.bse[j]*scale,
                     't':robust.tvalues[j],'p':robust.pvalues[j],
                     'ci95_B':ci[j].tolist(),'ci95_beta':(ci[j]*scale).tolist(),
                     'HC3_se':hc3.bse[j],'HC3_p':hc3.pvalues[j],
                     'HC3_ci95_B':ci3[j].tolist(),'HC3_ci95_beta':(ci3[j]*scale).tolist()})
    # Independent CRV1 calculation verifies the library covariance and t reference.
    xa=x.to_numpy();u=fit.resid.to_numpy();g=data.group.to_numpy();groups=np.unique(g)
    scores=np.array([(xa[g==v]*u[g==v,None]).sum(axis=0) for v in groups])
    bread=np.linalg.inv(xa.T@xa);correction=len(groups)/(len(groups)-1)*(len(data)-1)/(len(data)-x.shape[1])
    manual=correction*bread@(scores.T@scores)@bread
    cov_error=float(np.max(np.abs(manual-robust.cov_params())))
    assert np.allclose(manual,robust.cov_params(),rtol=1e-6,atol=1e-10)
    assert np.allclose(robust.pvalues,2*stats.t.sf(abs(robust.tvalues),len(groups)-1),rtol=1e-10,atol=1e-14)
    record={'n':len(data),'clusters':len(groups),'predictors':predictors,'parameters':x.shape[1],
            'r2':fit.rsquared,'adjusted_r2':fit.rsquared_adj,'residual_df':fit.df_resid,
            'cluster_test_df':len(groups)-1,'CRV1_multiplier':correction,
            'manual_cluster_covariance_max_difference':cov_error,'coefficients':rows}
    return record,fit,robust


def joint_test(robust,terms):
    names=robust.model.exog_names;L=np.zeros((len(terms),len(names)))
    for i,t in enumerate(terms):L[i,names.index(t)]=1
    f=robust.f_test(L)
    return {'terms':terms,'F':float(f.fvalue),'df_num':float(f.df_num),'df_den':float(f.df_denom),'p':float(f.pvalue)}


def diagnostics(fit):
    sw=stats.shapiro(fit.resid)
    bp=het_breuschpagan(fit.resid,fit.model.exog)
    vifs=[{'predictor':name,'VIF':float(variance_inflation_factor(fit.model.exog,i))}
          for i,name in enumerate(fit.model.exog_names) if name!='const']
    result={'shapiro_W':float(sw.statistic),'shapiro_p':float(sw.pvalue),
            'breusch_pagan_LM':float(bp[0]),'breusch_pagan_p':float(bp[1]),
            'VIF':vifs,'VIF_max':max(row['VIF'] for row in vifs),
            'collinearity_handling':'VIFs describe the specified equation; no automatic predictor selection.',
            'independence':'Community clustering; no temporal-order diagnostic'}
    result['interpretation']='Descriptive diagnostics only. Planned CRV1/HC3/bootstrap inference retained; no automatic variable or case removal.'
    result.pop('issues',None)
    influence=fit.get_influence()
    result['leverage_max']=float(influence.hat_matrix_diag.max())
    result['leverage_above_2k_n_count']=int((influence.hat_matrix_diag>2*fit.model.exog.shape[1]/fit.nobs).sum())
    result['cooks_distance_max']=float(influence.cooks_distance[0].max())
    result['cooks_above_4_n_count']=int((influence.cooks_distance[0]>4/fit.nobs).sum())
    return result


def path_points(arr,p,c):
    """Ten statistics per exposure; use each resample's SDs, not original SDs."""
    x=np.column_stack([np.ones(len(arr)),arr[:,:p+c]])
    med=arr[:,-3:-1]
    a,_,rank,_=np.linalg.lstsq(x,med,rcond=None)
    if rank<x.shape[1]:raise ValueError('rank-deficient activity resample')
    z=np.column_stack([x,med])
    b,_,rank,_=np.linalg.lstsq(z,arr[:,-1],rcond=None)
    if rank<z.shape[1]:raise ValueError('rank-deficient outcome resample')
    indirect=a[1:p+1]*b[-2:];direct=b[1:p+1]
    raw=np.column_stack([indirect,indirect.sum(axis=1),direct,direct+indirect.sum(axis=1)])
    sy=arr[:,-1].std(ddof=1);sx=arr[:,:p].std(axis=0,ddof=1)
    if sy<=0 or (sx<=0).any():raise ValueError('zero outcome or exposure SD')
    result=np.column_stack([raw,raw*(sx/sy)[:,None]]).ravel()
    if not np.isfinite(result).all():raise ValueError('non-finite path statistics')
    return result


def bca_interval(boot,point,jack):
    prop=((boot<point).sum()+.5*(boot==point).sum())/len(boot)
    bias=stats.norm.ppf(np.clip(prop,.5/len(boot),1-.5/len(boot)))
    diff=jack.mean()-jack;denom=6*np.sum(diff**2)**1.5
    acceleration=np.sum(diff**3)/denom if denom>0 else 0.
    z=stats.norm.ppf([.025,.975]);den=1-acceleration*(bias+z)
    adj=stats.norm.cdf(bias+(bias+z)/den)
    if np.any(den<=0) or adj[0]>=adj[1]:
        return {'status':'unstable_BCa_mapping','ci':None,'z0':bias,'acceleration':acceleration,'adjusted_quantiles':adj}
    return {'status':'ok','ci':np.quantile(boot,adj).tolist(),'z0':bias,'acceleration':acceleration,
            'adjusted_quantiles':adj.tolist(),'tail_draws_min':float(min(adj[0],1-adj[1])*len(boot))}


def bootstrap_paths(spec,n=N_BOOT,seed=SEED,save=True):
    d=spec['data'];ex=spec['exposures'];cov=spec['covariates'];p=len(ex);c=len(cov)
    arr=d[ex+cov+['Outdoor','Screen','VA']].to_numpy(float)
    groups=[np.flatnonzero(d.group.to_numpy()==g) for g in d.group.unique()]
    point=path_points(arr,p,c);rng=np.random.default_rng(seed)
    values=[];fail=[];draw_ids=[];draw_ns=[]
    for k in range(n):
        sample=rng.integers(0,len(groups),len(groups))
        idx=np.concatenate([groups[j] for j in sample])
        try:
            values.append(path_points(arr[idx],p,c));draw_ids.append(k);draw_ns.append(len(idx))
        except (ValueError,np.linalg.LinAlgError) as exc:fail.append({'draw_zero_based':k,'n':len(idx),'reason':str(exc)})
        if (k+1)%1000==0:log(f"{spec['name']}: bootstrap {k+1}/{n}, successful {len(values)}")
    values=np.asarray(values)
    if len(values)<2:raise RuntimeError('Insufficient estimable bootstrap draws')
    jack=[];jfail=[]
    for i,g in enumerate(groups):
        try:jack.append(path_points(np.delete(arr,g,axis=0),p,c))
        except (ValueError,np.linalg.LinAlgError) as exc:jfail.append({'cluster_index':i,'reason':str(exc)})
    jack=np.asarray(jack)
    kinds=['outdoor_B','screen_B','total_indirect_B','direct_B','total_B','outdoor_beta','screen_beta','total_indirect_beta','direct_beta','total_beta']
    labels=[e+'_'+kind for e in ex for kind in kinds];records={}
    for j,label in enumerate(labels):
        bca=bca_interval(values[:,j],point[j],jack[:,j]) if not jfail else {'status':'unavailable_incomplete_cluster_jackknife','ci':None}
        records[label]={'point':point[j],'ci95':np.quantile(values[:,j],[.025,.975]).tolist(),
                        'bca95':bca,'bootstrap_mean':values[:,j].mean(),'bootstrap_se':values[:,j].std(ddof=1)}
    if save:
        folder=OUTPUT/'bootstrap_LOCAL_ONLY';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(spec['name']+'.npz'),draws=values,labels=np.array(labels),point=point,
                            successful_draw_ids=draw_ids,resampled_n=draw_ns,jackknife=jack,seed=seed,attempted=n)
    return {'kind':'whole_cluster','seed':seed,'attempted':n,'successful':len(values),'failed':fail,
            'jackknife_attempted':len(groups),'jackknife_successful':len(jack),'jackknife_failed':jfail,
            'resampled_n_min':min(draw_ns),'resampled_n_max':max(draw_ns),'n':len(arr),'clusters':len(groups),
            'interval_scope':'Pointwise only; not family-wise or causal evidence.',
            'results':records}


def sem_crosscheck(spec,paths):
    d=spec['data'];exog=spec['exposures']+spec['covariates'];cols=exog+['Outdoor','Screen','VA']
    z=(d[cols]-d[cols].mean())/d[cols].std(ddof=1)
    syntax='Outdoor ~ '+' + '.join(exog)+'\nScreen ~ '+' + '.join(exog)+'\nVA ~ '+' + '.join(exog+['Outdoor','Screen'])+'\nOutdoor ~~ Screen\n'
    for i,x in enumerate(exog):syntax+=x+' ~~ '+' + '.join(exog[i:])+'\n'
    model=semopy.Model(syntax)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit=model.fit(z,options={'ftol':1e-12,'maxiter':5000});ins=model.inspect()
    rows=[]
    for y,record in paths.items():
        for r in record['coefficients']:
            x=r['predictor']
            if x=='const':continue
            value=float(ins[(ins.lval==y)&(ins.rval==x)&(ins.op=='~')].Estimate.iloc[0])
            rows.append({'outcome':y,'predictor':x,'OLS_beta':r['beta'],'SEM_beta':value,'absolute_difference':abs(r['beta']-value)})
    # Count covariance moments and free parameters directly; no meaningless df=0 fit indices.
    moments=len(cols)*(len(cols)+1)//2;free=len(model.param_vals);df=moments-free
    error=max(r['absolute_difference'] for r in rows)
    assert fit.success and df==0 and error<2e-5,(spec['name'],fit.success,df,error)
    return {'converged':bool(fit.success),'message':str(fit.message),'objective':'MLW','df':df,
            'covariance_moments':moments,'free_parameters':free,'specification':syntax,
            'max_difference':error,'tolerance':2e-5,'coefficients':rows,
            'warnings':[str(w.message) for w in caught],
            'scope':'Point-estimate algebra/numerical cross-check only; saturation is not evidence of model fit.'}


def fit_linear(spec,n_boot=N_BOOT):
    d=spec['data'];ex=spec['exposures'];cov=spec['covariates'];name=spec['name']
    out={'name':name,'n':len(d),'clusters':d.group.nunique(),'exposures':ex,'covariates':cov,
         'coding':spec['coding'],'paths':{},'joint_domains':[],'diagnostics':{}}
    fits={}
    log(f'{name}: fitting three equations')
    for y in ['Outdoor','Screen','VA']:
        predictors=ex+cov+(['Outdoor','Screen'] if y=='VA' else [])
        rec,fit,robust=ols(d,y,predictors);out['paths'][y]=rec;fits[y]=fit
        out['diagnostics'][y]=diagnostics(fit)
        if y!='VA':
            for domain,terms in spec['domains'].items():
                assert terms and all(t in predictors for t in terms)
                out['joint_domains'].append({'outcome':y,'domain':domain,**joint_test(robust,terms)})
    activity=[r for y in ['Outdoor','Screen'] for r in out['paths'][y]['coefficients'] if r['predictor'] in ex]
    activity += [r for r in out['paths']['VA']['coefficients'] if r['predictor'] in ['Outdoor','Screen']]
    direct=[r for r in out['paths']['VA']['coefficients'] if r['predictor'] in ex]
    holm(activity,'p_holm_activity_family');holm(direct,'p_holm_direct_family');holm(out['joint_domains'],'p_holm_domain_family')
    assert len(activity)==2*len(ex)+2 and len(direct)==len(ex) and len(out['joint_domains'])==14
    by={y:{r['predictor']:r for r in rec['coefficients']} for y,rec in out['paths'].items()}
    joint=[]
    for e in ex:
        for m in ['Outdoor','Screen']:
            pa=by[m][e]['p'];pb=by['VA'][m]['p']
            joint.append({'exposure':e,'via':m,'p_a':pa,'p_b':pb,'p':max(pa,pb)})
    holm(joint,'p_holm_specific_indirect_family')
    for row in joint:row['meets_joint_Holm_0_05']=row['p_holm_specific_indirect_family']<.05
    out['specific_indirect_tests']=joint
    out['multiplicity']={'activity_paths':len(activity),'direct_paths':len(direct),'domain_tests':14,'specific_indirect_paths':len(joint)}
    out['mediator_residual_correlation']=float(np.corrcoef(fits['Outdoor'].resid,fits['Screen'].resid)[0,1])
    point=path_points(d[ex+cov+['Outdoor','Screen','VA']].to_numpy(float),len(ex),len(cov)).reshape(len(ex),10)
    reduced=sm.OLS(d.VA,sm.add_constant(d[ex+cov],has_constant='add')).fit()
    err=float(np.max(np.abs(reduced.params[ex].to_numpy()-point[:,4])))
    assert err<1e-10
    out['total_algebra_max_difference']=err
    if name in ['subjective','contextual']:out['sem_crosscheck']=sem_crosscheck(spec,out['paths'])
    out['indirect']=bootstrap_paths(spec,n=n_boot)
    save_json(f'models/{name}.json',out)
    log(f"{name}: saved; bootstrap {out['indirect']['successful']}/{n_boot}")
    return out
