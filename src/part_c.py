"""Part C: privacy-preserving, guarded and budget-aware support ticket triage."""
from pathlib import Path
import json, re, random, time
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ALLOWED={'billing','delivery','technical','fraud'}; MAX_INPUT_TOKENS=400; BASE_DELAY=.05
ZERO_SYSTEM='CLASSIFY. Classify the ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}. Do not add prose.'
FEW_SYSTEM='CLASSIFY. Classify the ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}. Do not add prose.\nEXAMPLES:\nTicket: Someone hacked my account and made an unauthorized transaction.\n{"category":"fraud","reason":"unauthorized activity"}\nTicket: The invoice shows an extra charge.\n{"category":"billing","reason":"billing discrepancy"}'

def redact(text):
    counts={}
    for name,pat,rep in [('emails',EMAIL,'[EMAIL]'),('phones',PHONE,'[PHONE]'),('order_ids',ORDER,'[ORDER_ID]')]:
        text,n=pat.subn(rep,text); counts[name]=n
    return text,counts

def classify_run(prompt_kind='few', token_budget=None, output_name=None):
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    segments=pd.read_csv(OUT/'customer_segments.csv')
    system=FEW_SYSTEM if prompt_kind=='few' else ZERO_SYSTEM
    mock_llm.reset(); rng=random.Random(42); running_tokens=0; all_counts={'emails':0,'phones':0,'order_ids':0}; rows=[]; confusion={}; retry_lines=[]
    for _,t in tickets.iterrows():
        tid=t.ticket_id; original=t.text; redacted,counts=redact(original)
        for k,v in counts.items(): all_counts[k]+=v
        assert not EMAIL.search(redacted) and not PHONE.search(redacted) and not ORDER.search(redacted), f'PII remains in {tid}'
        row={'ticket_id':tid,'customer_id':t.customer_id,'status':'','category':'','segment_name':'','priority':'NORMAL','route':'human_review','tokens_used':0,'true_category':t.true_category}
        if not original.strip(): row['status']='skipped_empty'
        elif mock_llm.est_tokens(original)>MAX_INPUT_TOKENS: row['status']='flagged_long'
        elif token_budget is not None and running_tokens>=token_budget: row['status']='budget_exhausted'
        else:
            messages=[{'role':'system','content':system},{'role':'user','content':redacted}]
            response=None
            for retry in range(4):
                try:
                    assert messages[-1]['content']==redacted
                    response=mock_llm.chat(messages); break
                except mock_llm.TransientLLMError:
                    if retry>=3: raise
                    wait=BASE_DELAY*(2**retry)*rng.uniform(.9,1.15)
                    retry_lines.append(f'{tid} retry={retry} wait={wait:.4f}s')
                    time.sleep(wait)
            usage=int(response.usage.total_tokens); running_tokens+=usage; row['tokens_used']=usage
            raw=response.choices[0].message.content or ''
            try:
                parsed=json.loads(raw)
                if isinstance(parsed,dict) and parsed.get('category') in ALLOWED:
                    row['status']='classified_json'; row['category']=parsed['category']
                else: row['status']='flagged_invalid_output'
            except (json.JSONDecodeError,TypeError):
                words=re.findall(r'[a-z]+',raw.lower()); matches=set(words)&ALLOWED
                if len(matches)==1:
                    row['status']='classified_keyword'; row['category']=next(iter(matches))
                else: row['status']='flagged_invalid_output'
            if row['status'] in ('classified_json','classified_keyword'):
                confusion[(t.true_category,row['category'])]=confusion.get((t.true_category,row['category']),0)+1
        seg=segments[segments.customer_id.eq(t.customer_id)]
        if len(seg): row['segment_name']=seg.iloc[0].segment_name
        if row['status'] in ('skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'):
            row['route']='human_review'; row['priority']='NORMAL'
        elif row['category']=='fraud': row['route']='security_team'; row['priority']='HIGH'
        elif len(seg) and bool(seg.iloc[0].is_premium): row['route']='premium_desk'; row['priority']='HIGH'
        else: row['route']='support_queue'; row['priority']='NORMAL'
        rows.append(row)
    df=pd.DataFrame(rows)
    if output_name: df.drop(columns=['true_category']).to_csv(OUT/output_name,index=False)
    eligible=df[df.status.isin(['classified_json','classified_keyword'])]
    accuracy=float((eligible.category==df.loc[eligible.index,'true_category']).mean()) if len(eligible) else 0
    counts_status=df.status.value_counts().to_dict()
    summary={'prompt_kind':prompt_kind,'total_tickets':len(df),'status_counts':counts_status,'classified':len(eligible),'accuracy_classified':accuracy,'tokens_total':int(running_tokens),'tokens_per_classified':round(running_tokens/max(len(eligible),1),2),'redactions':all_counts,'redactions_total':sum(all_counts.values()),'remaining_pii_matches':0,'budget':token_budget,'budget_exhausted':int((df.status=='budget_exhausted').sum()),'confusion':{f'{a}->{b}':n for (a,b),n in confusion.items()},'retry_count':len(retry_lines),'retry_log':retry_lines}
    return df,summary

def run():
    zero,zs=classify_run('zero',None,'triage_zero_shot.csv')
    few,fs=classify_run('few',None,'triage_few_shot.csv')
    budget_df,bs=classify_run('zero',6000,'triage_budget_demo.csv')
    if bs['budget_exhausted']==0:
        budget_df,bs=classify_run('zero',5000,'triage_budget_demo.csv')
    final,finals=classify_run('few',100000,'triage_results.csv')
    (OUT/'part_c_summary.json').write_text(json.dumps({'zero_shot':zs,'few_shot':fs,'budget_demo':bs,'final_run':finals},indent=2),encoding='utf-8')
    (OUT/'triage_retry_log.txt').write_text('\n'.join(fs['retry_log']),encoding='utf-8')
    print('Part C triage complete:',fs['status_counts'])
    return {'zero_shot':zs,'few_shot':fs,'budget_demo':bs,'final_run':finals}
if __name__=='__main__': run()
