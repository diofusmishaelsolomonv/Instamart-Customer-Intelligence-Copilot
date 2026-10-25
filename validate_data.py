"""Independent checkpoint validation for the Part A cleaned dataset."""
from pathlib import Path
import json, sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; CLEAN=ROOT/'data/clean'; DB=ROOT/'data/instamart.db'
def main():
    c=pd.read_csv(CLEAN/'customers.csv'); p=pd.read_csv(CLEAN/'products.csv'); o=pd.read_csv(CLEAN/'orders.csv')
    checks=[]
    checks.append(len(c)==1200); checks.append(len(p)==40); checks.append(len(o)==12309)
    checks.append(o.order_id.nunique()==12309); checks.append((o.status=='completed').sum()==11110)
    checks.append((o.status=='cancelled').sum()==1199); checks.append(not (o.status=='pending').any())
    checks.append(o.order_amount.notna().all()); checks.append(o.loc[o.status=='completed','delivery_minutes'].notna().all())
    checks.append(o.loc[o.status=='cancelled','delivery_minutes'].isna().all())
    checks.append(abs(o.loc[o.status=='completed','order_amount'].sum()-2998668.80)<.01)
    checks.append(c.age.notna().all() and c.age.between(18,60).all())
    checks.append(c.city.nunique()==6 and set(c.gender.unique())=={'female','male','unknown'})
    with sqlite3.connect(DB) as conn:
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    checks.append({'customers','products','orders'}.issubset(tables))
    assert len(checks)>=8 and all(checks), f'Validation failed: {checks}'
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
