"""Part B: RFM segmentation and leakage-conscious lapse modelling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAPSHOT=pd.Timestamp('2025-10-01 00:00:00'); CUTOFF=pd.Timestamp('2025-09-01 00:00:00')

def build_rfm(orders, customers):
    o=orders.copy(); o['order_ts']=pd.to_datetime(o.order_ts); o=o[o.status.eq('completed')]
    r=o.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    r['recency']=(SNAPSHOT-r.last_order_ts).dt.days
    r=r[['customer_id','recency','frequency','monetary']]
    scaler=StandardScaler(); scaled=scaler.fit_transform(r[['recency','frequency','monetary']])
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(scaled)
        from sklearn.metrics import silhouette_score
        rows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(scaled,km.labels_)})
    pd.DataFrame(rows).to_csv(OUT/'kmeans_metrics.csv',index=False)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(scaled); r['segment_id']=km.labels_
    profiles=r.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    order=profiles.sort_values('mean_monetary',ascending=False).segment_id.tolist()
    names={order[0]:'Premium',order[1]:'Frequent Essentials',order[2]:'Occasional Essentials',order[3]:'At-Risk / Low-Activity'}
    actions={'Premium':'Offer relevant cross-category bundles and protect service quality.', 'Frequent Essentials':'Encourage repeat baskets with low-friction loyalty rewards.', 'Occasional Essentials':'Use personalised replenishment reminders and targeted offers.', 'At-Risk / Low-Activity':'Run a low-cost win-back campaign and monitor response.'}
    profiles['segment_name']=profiles.segment_id.map(names); profiles['recommended_action']=profiles.segment_name.map(actions)
    profiles['is_premium']=profiles.segment_name.eq('Premium')
    profiles.to_csv(OUT/'segment_profiles.csv',index=False)
    r['segment_name']=r.segment_id.map(names); r['is_premium']=r.segment_name.eq('Premium')
    pca=PCA(n_components=2).fit(scaled)
    r[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
    (OUT/'rfm_summary.json').write_text(json.dumps({'customers':len(r),'recency_median':float(r.recency.median()),'recency_min':int(r.recency.min()),'recency_max':int(r.recency.max()),'frequency_mean':float(r.frequency.mean()),'frequency_max':int(r.frequency.max()),'monetary_mean':float(r.monetary.mean()),'monetary_max':float(r.monetary.max()),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'profiles':profiles.to_dict('records')},indent=2,default=str),encoding='utf-8')
    return r

def build_lapse(orders, customers, segments):
    o=orders.copy(); o['order_ts']=pd.to_datetime(o.order_ts); c=customers.copy(); c['signup_date']=pd.to_datetime(c.signup_date)
    pre=o[(o.order_ts<CUTOFF)&o.status.isin(['completed','cancelled'])].copy()
    completed=pre[pre.status.eq('completed')]
    cohort=completed.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index()
    cohort['recency']=(CUTOFF-cohort.last_order).dt.days
    totals=pre.groupby('customer_id').size().rename('all_final')
    cancels=pre[pre.status.eq('cancelled')].groupby('customer_id').size().rename('cancelled')
    cohort=cohort.merge(totals,on='customer_id',how='left').merge(cancels,on='customer_id',how='left')
    cohort['cancellation_rate']=cohort.cancelled.fillna(0)/cohort.all_final
    fav=completed.merge(pd.read_csv(CLEAN/'products.csv')[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda x:x.value_counts().index[0]).rename('favourite_category')
    cohort=cohort.merge(fav,on='customer_id',how='left').merge(c,on='customer_id',how='left')
    cohort['tenure_days']=(CUTOFF-cohort.signup_date).dt.days
    post=o[(o.order_ts>=CUTOFF)&(o.order_ts<pd.Timestamp('2025-10-01'))&o.status.eq('completed')].groupby('customer_id').order_id.nunique()
    cohort['lapsed']=(~cohort.customer_id.isin(post.index)).astype(int)
    numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']
    categorical=['city','gender','favourite_category']
    for col in numeric: cohort[col]=pd.to_numeric(cohort[col],errors='coerce')
    cohort[numeric]=cohort[numeric].replace([np.inf,-np.inf],np.nan)
    for col in categorical: cohort[col]=cohort[col].fillna('unknown').astype(str)
    X=cohort[numeric+categorical]; y=cohort.lapsed
    assert (pre.order_ts < CUTOFF).all(), 'Post-cutoff order leaked into features'
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    preprocessor=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='unknown')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))]),categorical)])
    baseline=np.zeros(len(y_test),dtype=int)
    baseline_metrics={'accuracy':accuracy_score(y_test,baseline),'precision':precision_score(y_test,baseline,zero_division=0),'recall':recall_score(y_test,baseline,zero_division=0),'f1':f1_score(y_test,baseline,zero_division=0)}
    lr=Pipeline([('prep',preprocessor),('model',LogisticRegression(max_iter=1000,random_state=42))])
    lrg=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); lrg.fit(X_train,y_train)
    rf=Pipeline([('prep',preprocessor),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rfg=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); rfg.fit(X_train,y_train)
    def metrics(model):
        pred=model.predict(X_test)
        return {'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),'accuracy':accuracy_score(y_test,pred)}
    chosen=rfg.best_estimator_ if metrics(rfg.best_estimator_)['f1']>=metrics(lrg.best_estimator_)['f1'] else lrg.best_estimator_
    selected_name='random_forest' if chosen is rfg.best_estimator_ else 'logistic_regression'
    joblib.dump(chosen,OUT/'lapse_model.joblib')
    loaded=joblib.load(OUT/'lapse_model.joblib'); predictions_equal=bool(np.array_equal(chosen.predict(X_test),loaded.predict(X_test)))
    n0=int((y_train==0).sum()); n1=int((y_train==1).sum()); minority=X_train[y_train.eq(1)].copy()
    if n1<n0:
        sampled=minority.sample(n=n0-n1,replace=True,random_state=42)
        X_bal=pd.concat([X_train,sampled],ignore_index=True); y_bal=pd.concat([y_train.reset_index(drop=True),pd.Series(np.ones(len(sampled),dtype=int))],ignore_index=True)
    else: X_bal=X_train; y_bal=y_train
    balanced_lr=Pipeline([('prep',preprocessor),('model',LogisticRegression(max_iter=1000,random_state=42,C=lrg.best_params_['model__C']))]); balanced_lr.fit(X_bal,y_bal)
    rf_model=rfg.best_estimator_; prep=rf_model.named_steps['prep']; feature_names=prep.get_feature_names_out()
    importances=getattr(rf_model.named_steps['model'],'feature_importances_',np.array([]))
    top_features=sorted([{'feature':str(n),'importance':float(v)} for n,v in zip(feature_names,importances)],key=lambda z:z['importance'],reverse=True)[:5]
    lr_model=lrg.best_estimator_; lr_names=lr_model.named_steps['prep'].get_feature_names_out(); coefs=lr_model.named_steps['model'].coef_[0]
    coeffs=[{'feature':str(n),'coefficient':float(v),'sign':'positive' if v>0 else 'negative' if v<0 else 'zero'} for n,v in zip(lr_names,coefs)]
    test_ids=cohort.loc[X_test.index,['customer_id','lapsed']].copy(); probs=chosen.predict_proba(X_test)[:,list(chosen.classes_).index(1)]
    test_ids['lapse_probability']=probs; test_ids=test_ids.merge(segments[['customer_id','is_premium']],on='customer_id',how='left'); ret=test_ids[test_ids.is_premium.eq(True)].sort_values('lapse_probability',ascending=False).head(10)
    ret.rename(columns={'lapsed':'actually_lapsed'})[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT/'retention_list.csv',index=False)
    result={'cohort_size':len(cohort),'lapsed_count':int(y.sum()),'lapsed_rate':float(y.mean()),'train_rows':len(X_train),'test_rows':len(X_test),'baseline':baseline_metrics,'logistic_best_params':lrg.best_params_,'logistic_best_cv_f1':lrg.best_score_,'logistic_test':metrics(lrg.best_estimator_),'random_forest_best_params':rfg.best_params_,'random_forest_best_cv_f1':rfg.best_score_,'random_forest_test':metrics(rfg.best_estimator_),'selected_model':selected_name,'reloaded_predictions_equal':predictions_equal,'train_scaler_frequency_mean':float(lrg.best_estimator_.named_steps['prep'].named_transformers_['num'].named_steps['scale'].mean_[numeric.index('frequency')]),'full_cohort_frequency_mean':float(cohort.frequency.mean()),'balanced_logistic_test':metrics(balanced_lr),'random_forest_top_features':top_features,'logistic_coefficients':sorted(coeffs,key=lambda z:abs(z['coefficient']),reverse=True)[:10],'premium_retention_list_true_lapsed':int(ret.lapsed.sum()) if len(ret) else 0}
    (OUT/'lapse_model_summary.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return result

def run():
    orders=pd.read_csv(CLEAN/'orders.csv'); customers=pd.read_csv(CLEAN/'customers.csv')
    segments=build_rfm(orders,customers); lapse=build_lapse(orders,customers,segments)
    print('Part B segmentation and lapse modelling complete.')
    return {'rfm':json.loads((OUT/'rfm_summary.json').read_text()),'lapse':lapse}
if __name__=='__main__': run()
