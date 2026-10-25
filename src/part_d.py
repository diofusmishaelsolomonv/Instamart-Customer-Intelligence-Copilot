"""Part D: retrieval-grounded policy assistant and resumable tool-calling refund agent."""
from pathlib import Path
import json, re, time, uuid, sqlite3, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; POL=ROOT/'data/policies'; RAW=ROOT/'data/raw'; OUT=ROOT/'outputs'; STATE=ROOT/'state'; DB=ROOT/'data/instamart.db'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
DOCS=['refund_policy.txt','delivery_policy.txt','payment_policy.txt','privacy_policy.txt','escalation_policy.txt']

def chunk_documents(overlap=10):
    chunks=[]; step=40-overlap
    for fn in DOCS:
        words=(POL/fn).read_text(encoding='utf-8').split()
        for start in range(0,len(words),step):
            end=min(start+40,len(words))
            chunks.append({'source':fn,'text':' '.join(words[start:end])})
            if end>=len(words): break
    return chunks

def build_retriever(overlap=10):
    chunks=chunk_documents(overlap)
    vectorizer=TfidfVectorizer(); matrix=vectorizer.fit_transform([c['text'] for c in chunks])
    def retrieve(query,k=3):
        q=vectorizer.transform([query]); scores=cosine_similarity(q,matrix).ravel(); idx=scores.argsort()[::-1][:k]
        return [{'source':chunks[i]['source'],'score':float(scores[i]),'text':chunks[i]['text']} for i in idx]
    return chunks,retrieve

CHUNKS, retrieve=build_retriever(10)
def answer(question, retriever=retrieve):
    results=retriever(question,3)
    context='\n'.join(f"[{r['source']}] {r['text']}" for r in results)
    messages=[{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say exactly that you could not find the information in the policy documents.'},
              {'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
    response=mock_llm.chat(messages)
    return {'answer':response.choices[0].message.content,'sources':list(dict.fromkeys(r['source'] for r in results)),'retrieved':results}

def evaluate_rag():
    ev=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False); rows=[]
    for _,r in ev.iterrows():
        result=answer(r.question); sources=result['sources']; expected_doc=r.expected_doc; expected_fact=r.expected_fact
        rows.append({'question':r.question,'expected_doc':expected_doc,'top3_sources':'|'.join(sources),'doc_hit':expected_doc in sources if expected_doc!='NONE' else False,'expected_fact_hit':expected_fact in result['answer'] if expected_fact else False,'fallback':result['answer'].startswith('I could not find')})
    df=pd.DataFrame(rows); df.to_csv(OUT/'policy_eval_results.csv',index=False)
    zero_chunks,zero_retrieve=build_retriever(0); zero_rows=[]
    for _,r in ev.iterrows():
        result=answer(r.question,zero_retrieve)
        zero_rows.append({'question':r.question,'expected_doc':r.expected_doc,'top3_sources':'|'.join(result['sources']),'doc_hit':r.expected_doc in result['sources'] if r.expected_doc!='NONE' else False,'expected_fact_hit':r.expected_fact in result['answer'] if r.expected_fact else False,'fallback':result['answer'].startswith('I could not find')})
    zdf=pd.DataFrame(zero_rows); zdf.to_csv(OUT/'policy_eval_zero_overlap.csv',index=False)
    summary={'chunk_count':len(CHUNKS),'chunk_counts_by_source':pd.Series([c['source'] for c in CHUNKS]).value_counts().to_dict(),'overlap_10':{'answerable_doc_hits':int(df[df.expected_doc!='NONE'].doc_hit.sum()),'answerable_fact_hits':int(df[df.expected_doc!='NONE'].expected_fact_hit.sum()),'unanswerable_fallbacks':int(df[df.expected_doc=='NONE'].fallback.sum())},'overlap_0':{'answerable_doc_hits':int(zdf[zdf.expected_doc!='NONE'].doc_hit.sum()),'answerable_fact_hits':int(zdf[zdf.expected_doc!='NONE'].expected_fact_hit.sum()),'unanswerable_fallbacks':int(zdf[zdf.expected_doc=='NONE'].fallback.sum()),'failed_questions':zdf[(zdf.expected_doc!='NONE')&(~zdf.expected_fact_hit)].question.tolist()}}
    return summary

# Tools use deterministic synthetic records only. No real payments are issued.
def get_order(order_id):
    with sqlite3.connect(DB) as conn:
        row=conn.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    if not row: return {'error':'order not found'}
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else int(row[3])}
def search_policy(query): return {'results':retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount); mins=delivery_minutes
    return {'refund_inr':round(amount,2) if mins is not None and int(mins)>45 else 0.0}
def _ledger(tool_name,order_id,refund_inr):
    record={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOLS_SPEC=[{'name':'get_order','description':'Look up one order from the local SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},{'name':'search_policy','description':'Retrieve policy chunks relevant to a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},{'name':'calculate_refund','description':'Calculate refund eligibility from order amount and delivery minutes.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':['integer','null']}},'required':['order_amount','delivery_minutes']}},{'name':'issue_refund','description':'Record an eligible automatic refund in the local ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},{'name':'escalate_to_human','description':'Record a refund request requiring human lead approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}]
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

class SimulatedCrash(RuntimeError): pass

def _save_checkpoint(run_id,scenario_id,messages,tokens_used,step,tools_executed,user_message,steps=None,llm_calls=0):
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step_number':step,'tools_executed':tools_executed,'user_message':user_message,'steps':steps or [],'llm_calls':llm_calls}
    (STATE/f'{run_id}.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

def _status(answer):
    if not answer: return 'FALLBACK'
    return answer.split(':',1)[0].strip().upper()

def _continue_agent(data,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tools=None):
    start=time.monotonic(); messages=data['messages']; tokens_used=int(data['tokens_used']); steps=data.get('steps',[]); tools_executed=data.get('tools_executed',[]); llm_calls=int(data.get('llm_calls',0)); run_id=data['run_id']; scenario_id=data['scenario_id']; user_message=data['user_message']
    while llm_calls<max_iterations:
        if time.monotonic()-start>=time_limit_seconds:
            final='FALLBACK_TIMEOUT: hand off to a human agent; execution time limit reached.'
            return {'status':'FALLBACK_TIMEOUT','final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        response=mock_llm.chat(messages,tools=TOOLS_SPEC); llm_calls+=1; tokens_used+=int(response.usage.total_tokens)
        if tokens_used>token_limit:
            final='FALLBACK_TOKEN_LIMIT: hand off to a human agent; token budget exceeded before executing the requested tool.'
            return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        msg=response.choices[0].message; calls=getattr(msg,'tool_calls',[]) or []
        if not calls:
            final=msg.content or 'FALLBACK: empty assistant response; hand off to a human agent.'
            return {'status':_status(final),'final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        tc=calls[0]; name=tc.name; args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
        messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':name,'arguments':args}]})
        if name not in TOOL_MAP: observation={'error':f'unknown tool {name}'}
        else: observation=TOOL_MAP[name](**args); tools_executed.append(name)
        tool_msg={'role':'tool','tool_call_id':tc.id,'name':name,'content':json.dumps(observation,default=str)}
        messages.append(tool_msg); steps.append({'tool_name':name,'arguments':args,'observation':observation})
        data.update({'messages':messages,'tokens_used':tokens_used,'step_number':len(steps),'tools_executed':tools_executed,'llm_calls':llm_calls,'steps':steps})
        _save_checkpoint(run_id,scenario_id,messages,tokens_used,len(steps),tools_executed,user_message,steps,llm_calls)
        if crash_after_tools is not None and len(tools_executed)>=crash_after_tools:
            raise SimulatedCrash(f'Simulated crash after {len(tools_executed)} tool executions; checkpoint saved.')
    final='FALLBACK_MAX_ITERATIONS: hand off to a human agent; iteration limit reached.'
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    run_id=run_id or f'{scenario_id}_{uuid.uuid4().hex[:8]}'
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT. Follow the refund policy and use the provided tools. Never issue a refund unless the tool workflow permits it.'},{'role':'user','content':user_message}],'tokens_used':0,'step_number':0,'tools_executed':[],'steps':[],'llm_calls':0,'user_message':user_message}
    _save_checkpoint(run_id,scenario_id,data['messages'],0,0,[],user_message,[],0)
    return _continue_agent(data,max_iterations,token_limit,time_limit_seconds,crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    data=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    data.setdefault('steps',[]); data.setdefault('llm_calls',data.get('step_number',0)); data.setdefault('user_message',next((m['content'] for m in data['messages'] if m['role']=='user'),''))
    return _continue_agent(data,max_iterations,token_limit,time_limit_seconds)

def run_agents():
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False)
    for checkpoint in STATE.glob('*.json'):
        checkpoint.unlink()
    (OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8'); runs=[]
    for _,s in scenarios.iterrows():
        r=run_agent(s.message,s.scenario_id)
        refund=None
        for st in r['steps']:
            if st['tool_name'] in ('calculate_refund','issue_refund','escalate_to_human'): refund=st['observation'].get('refund_inr')
        record={'scenario_id':s.scenario_id,'status':r['status'],'final_answer':r['final_answer'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'refund_inr':refund,'steps':r['steps'],'tools_executed':r['tools_executed'],'expected_outcome':s.expected_outcome,'expected_refund_inr':float(s.expected_refund_inr)}
        runs.append(record)
    with (OUT/'agent_runs.jsonl').open('w',encoding='utf-8') as f:
        for r in runs: f.write(json.dumps(r,default=str)+'\n')
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,'S1_TOKEN_DEMO',token_limit=200,run_id='S1_TOKEN_DEMO')
    timeout_demo=run_agent(s1.message,'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='S1_TIMEOUT_DEMO')
    crash_id='S1_CRASH_RESUME'
    try: run_agent(s1.message,'S1_CRASH_RESUME',run_id=crash_id,crash_after_tools=2)
    except SimulatedCrash: pass
    resumed=resume_agent(crash_id)
    crash_summary={'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['tools_executed'],'llm_calls':resumed['llm_calls'],'tokens_used':resumed['tokens_used']}
    summary={'rag':evaluate_rag(),'scenarios':runs,'token_limit_demo':{'status':token_demo['status'],'final_answer':token_demo['final_answer']},'timeout_demo':{'status':timeout_demo['status'],'final_answer':timeout_demo['final_answer']},'crash_resume':crash_summary}
    (OUT/'part_d_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    return summary

def run():
    return run_agents()
if __name__=='__main__':
    s=run(); print('Part D complete:',s['token_limit_demo']['status'],s['timeout_demo']['status'],s['crash_resume']['status'])
