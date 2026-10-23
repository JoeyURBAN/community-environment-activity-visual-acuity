"""Supplementary age-referenced binary visual-acuity model."""
from runtime import np, OUTPUT, save_csv, save_json, log
from config import COVARIATES, N_BOOT, SEED
from data_preparation import load_data
from scipy.special import expit
from scipy.optimize import minimize
import warnings
from firthmodels import FirthLogisticRegression
from importlib.metadata import version

def firth_fit(X,y):
    assert np.linalg.matrix_rank(np.column_stack([X,np.ones(len(y))]))==X.shape[1]+1
    if len(np.unique(y))<2:raise ValueError('Only one outcome class in bootstrap draw')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        m=FirthLogisticRegression(backend='numpy',max_iter=300,gtol=1e-8,xtol=1e-8,max_halfstep=40).fit(X,y)
    if not m.converged_:raise ValueError('Firth optimisation did not converge')
    beta=np.r_[m.coef_,m.intercept_]
    if not np.isfinite(beta).all():raise ValueError('Nonfinite Firth coefficients')
    return beta,m,[str(w.message) for w in caught]

def objective_score(beta,X,y):
    eta=X@beta;p=expit(eta);w=p*(1-p);I=X.T@(w[:,None]*X)
    sign,ld=np.linalg.slogdet(I)
    if sign<=0:raise np.linalg.LinAlgError('Singular information')
    ll=np.sum(y*eta-np.logaddexp(0,eta))+.5*ld
    h=w*np.sum(X*(np.linalg.solve(I,X.T).T),axis=1)
    score=X.T@(y-p+h*(.5-p))
    return -ll,-score

def bfgs_firth_fit(X,y):
    """Numerical fallback for the same Jeffreys-penalised likelihood."""
    if len(np.unique(y))<2 or np.linalg.matrix_rank(X)<X.shape[1]:
        raise ValueError('Non-estimable bootstrap sample')
    opt=minimize(lambda b:objective_score(b,X,y),np.zeros(X.shape[1]),jac=True,
                 method='BFGS',options={'gtol':1e-9,'maxiter':2000})
    score=float(np.max(abs(objective_score(opt.x,X,y)[1])))
    if score>=1e-6 or not np.isfinite(opt.x).all():
        raise ValueError('Fallback optimiser score remains above tolerance')
    return opt.x,score,str(opt.message)

def run_age(s):
    # Exact separated 2x2 fixture: Firth equals adding 1/2 to each cell.
    Xfixture=np.array([[0.],[0.],[0.],[1.],[1.],[1.]])
    b,_,_=firth_fit(Xfixture,np.array([0,0,0,1,1,1]))
    target=np.array([2*np.log(7),-np.log(7)])
    assert np.max(abs(b-target))<1e-6
    cols=['Outdoor','Screen']+COVARIATES
    X=s[cols].to_numpy(float);y=s.BelowAgeReference.to_numpy(int)
    means=X.mean(0);sd=X.std(0,ddof=1);z=(X-means)/sd
    coef,m,warn=firth_fit(z,y)
    zz=np.column_stack([z,np.ones(len(z))])
    # Independent numerical optimisation of Jeffreys-penalised likelihood.
    check=minimize(lambda b:objective_score(b,zz,y),np.zeros(zz.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    err=float(np.max(abs(check.x-coef)))
    score=float(np.max(abs(objective_score(coef,zz,y)[1])))
    assert err<1e-5 and score<1e-6,(err,score,check.message)
    rawcoef=np.r_[coef[:-1]/sd,coef[-1]-np.sum(coef[:-1]*means/sd)]
    groups=[np.flatnonzero(s.group.to_numpy()==g) for g in s.group.unique()]
    rng=np.random.default_rng(SEED);values=[];fail=[];drawids=[];recovered=[]
    for i in range(N_BOOT):
        idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        try:
            bx,fit,ww=firth_fit(z[idx],y[idx])
            values.append(np.r_[bx[:-1]/sd,bx[-1]-np.sum(bx[:-1]*means/sd)]);drawids.append(i)
        except Exception as first_exc:
            try:
                bx,fallback_score,message=bfgs_firth_fit(np.column_stack([z[idx],np.ones(len(idx))]),y[idx])
                values.append(np.r_[bx[:-1]/sd,bx[-1]-np.sum(bx[:-1]*means/sd)]);drawids.append(i)
                recovered.append({'draw':i,'modified_score_max':fallback_score,'message':message,
                                  'initial_error':str(first_exc)})
            except Exception as second_exc:
                fail.append({'draw':i,'n':len(idx),'events':int(y[idx].sum()),
                             'initial_error':str(first_exc),'fallback_error':str(second_exc)})
        if (i+1)%500==0:log(f'Age-reference Firth bootstrap {i+1}/{N_BOOT}; successful {len(values)}')
    order=np.argsort(drawids);values=np.asarray(values)[order];drawids=np.asarray(drawids)[order]
    ci=np.quantile(values,[.025,.975],axis=0)
    rows=[]
    for j,v in enumerate(cols+['Intercept']):
        rows.append({'term':v,'log_OR':rawcoef[j],'OR_per_unit':np.exp(rawcoef[j]),'ci95_OR_low':np.exp(ci[0,j]),'ci95_OR_high':np.exp(ci[1,j]),'interpretation':'Odds of either eye below age reference; per score unit for activity; exploratory cluster-bootstrap interval'})
    save_csv('tables/age_reference_firth_OR.csv',rows)
    np.savez_compressed(OUTPUT/'bootstrap_LOCAL_ONLY/age_reference_firth.npz',coefficients=values,draw_ids=drawids,columns=np.array(cols+['Intercept']),point=rawcoef,seed=SEED)
    save_json('components/age_reference_firth.json',{'n':len(s),'clusters':len(groups),'events':int(y.sum()),'event_clusters':s.loc[s.BelowAgeReference==1,'group'].nunique(),'covariates':cols,'slopes':len(cols),'method':'firthmodels '+version('firthmodels')+'; Jeffreys penalty 0.5, numpy backend','converged':bool(m.converged_),'iterations':m.n_iter_,'bootstrap_attempted':N_BOOT,'bootstrap_successful':len(values),'bootstrap_failed':fail,'recovered_identical_draws':recovered,'recovery_note':'The fallback uses the same data, sampled clusters and Jeffreys-penalised likelihood; only the numerical optimiser differs.','seed':SEED,'coefficients':rows,'warnings':warn,'verification':{'exact_separation_fixture_max_error':float(np.max(abs(b-target))),'independent_BFGS_max_coefficient_error':err,'modified_score_max':score,'BFGS_message':str(check.message)},'interpretation':'Supplementary estimation only; current age proxies examination age; no diagnostic myopia, no mediation products, no new p-value hypothesis claims; 21 events do not become a large sample through penalisation'})


if __name__=='__main__':
    if (OUTPUT/'components/age_reference_firth.json').exists():
        raise RuntimeError('Existing Firth output found; use a fresh working copy to avoid overwriting completed results.')
    _,_,s=load_data(export=False)
    run_age(s)
