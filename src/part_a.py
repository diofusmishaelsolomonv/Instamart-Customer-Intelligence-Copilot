"""Part A: deterministic data audit, cleaning, SQLite loading and SQL analysis."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
DB = ROOT / "data" / "instamart.db"


def audit():
    reports = {}
    for name in ("customers", "products", "orders"):
        df = pd.read_csv(RAW / f"{name}.csv")
        reports[name] = {"shape": list(df.shape), "dtypes": {c: str(df[c].dtype) for c in df.columns},
                         "missing": {c: {"count": int(df[c].isna().sum()),
                                         "percent": round(100*df[c].isna().mean(), 2)} for c in df.columns}}
    orders = pd.read_csv(RAW/"orders.csv")
    customers = pd.read_csv(RAW/"customers.csv")
    audit = {
        "raw_orders_rows": len(orders), "exact_duplicate_rows": int(orders.duplicated().sum()),
        "pending_rows": int((orders.status == "pending").sum()), "distinct_order_ids": int(orders.order_id.nunique()),
        "non_numeric_amounts": int(pd.to_numeric(orders.order_amount, errors="coerce").isna().sum()),
        "negative_delivery_minutes": int(pd.to_numeric(orders.delivery_minutes, errors="coerce").lt(0).sum()),
        "blank_delivery_minutes": int((orders.delivery_minutes.astype(str).str.strip()=="").sum()),
        "blank_age": int((customers.age.astype(str).str.strip()=="").sum()),
        "ages_outside_10_100": int(pd.to_numeric(customers.age, errors="coerce").pipe(lambda x: ((x<10)|(x>100)).sum())),
        "raw_city_spellings": int(customers.city.nunique()),
        "raw_nonblank_gender_spellings": int(customers.loc[customers.gender.astype(str).str.strip()!="", "gender"].nunique()),
        "blank_gender": int((customers.gender.astype(str).str.strip()=="").sum())}
    pd.Series(audit, name="value").to_csv(OUT/"part_a_audit.csv", header=True)
    (OUT/"raw_audit.json").write_text(__import__('json').dumps(reports, indent=2), encoding="utf-8")
    return audit


def clean_and_store():
    CLEAN.mkdir(parents=True, exist_ok=True); OUT.mkdir(exist_ok=True)
    customers = pd.read_csv(RAW/"customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW/"products.csv")
    orders = pd.read_csv(RAW/"orders.csv", keep_default_na=False)
    raw_city = customers.city.astype(str)
    normalized_city = raw_city.str.strip().str.title()
    city_values_standardized = int((raw_city != normalized_city).sum())
    customers["city"] = normalized_city
    raw_gender = customers.gender.astype(str)
    normalized_gender = raw_gender.str.strip().str.lower().map({"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
    gender_values_standardized = int((raw_gender != normalized_gender).sum())
    customers["gender"] = normalized_gender
    customers["age"] = pd.to_numeric(customers.age, errors="coerce")
    invalid_ages = int(((customers.age < 10) | (customers.age > 100)).sum())
    customers.loc[~customers.age.between(10,100), "age"] = np.nan
    age_values_imputed = int(customers.age.isna().sum())
    customers["age"] = customers.age.fillna(customers.groupby("city").age.transform("median"))
    customers["signup_date"] = pd.to_datetime(customers.signup_date).dt.strftime("%Y-%m-%d")
    products["price_inr"] = pd.to_numeric(products.price_inr)
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    raw_order_rows = len(orders)
    orders = orders.drop_duplicates().copy()
    exact_duplicates_removed = raw_order_rows - len(orders)
    after_exact_dedup_rows = len(orders)
    orders = orders.sort_values(["order_ts"]).drop_duplicates("order_id", keep="last").copy()
    latest_record_rows_removed = after_exact_dedup_rows - len(orders)
    orders["quantity"] = pd.to_numeric(orders.quantity).astype(int)
    orders["discount_pct"] = pd.to_numeric(orders.discount_pct)
    orders["order_amount"] = pd.to_numeric(orders.order_amount, errors="coerce")
    amount_values_imputed = int(orders.order_amount.isna().sum())
    orders = orders.merge(products[["product_id","price_inr","category"]], on="product_id", how="left", validate="many_to_one")
    missing_amount = orders.order_amount.isna()
    orders.loc[missing_amount,"order_amount"] = (orders.loc[missing_amount,"price_inr"] * orders.loc[missing_amount,"quantity"] * (1-orders.loc[missing_amount,"discount_pct"]/100)).round(2)
    orders["delivery_minutes"] = pd.to_numeric(orders.delivery_minutes, errors="coerce")
    negative_delivery_values_cleared = int((orders.delivery_minutes < 0).sum())
    orders.loc[orders.delivery_minutes < 0,"delivery_minutes"] = np.nan
    completed_delivery_values_imputed = int((orders.status.eq("completed") & orders.delivery_minutes.isna()).sum())
    completed = orders.status.eq("completed")
    medians = orders.loc[completed].groupby("category").delivery_minutes.median()
    miss_completed = completed & orders.delivery_minutes.isna()
    orders.loc[miss_completed,"delivery_minutes"] = orders.loc[miss_completed,"category"].map(medians)
    orders = orders.drop(columns=["price_inr","category"])
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    for name, df in (("customers",customers),("products",products),("orders",orders)):
        df.to_csv(CLEAN/f"{name}.csv", index=False)
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        for name, df in (("customers",customers),("products",products),("orders",orders)):
            df.to_sql(name, conn, if_exists="replace", index=False)
    stats = {"customers":len(customers),"products":len(products),"orders":len(orders),
             "city_values_standardized":city_values_standardized,"gender_values_standardized":gender_values_standardized,
             "invalid_ages_outside_10_100":invalid_ages,"age_values_imputed":age_values_imputed,
             "exact_duplicates_removed":exact_duplicates_removed,"latest_record_rows_removed":latest_record_rows_removed,
             "amount_values_imputed":amount_values_imputed,"negative_delivery_values_cleared":negative_delivery_values_cleared,
             "completed_delivery_values_imputed":completed_delivery_values_imputed,
             "completed":int((orders.status=="completed").sum()),"cancelled":int((orders.status=="cancelled").sum()),
             "missing_amount":int(orders.order_amount.isna().sum()),
             "missing_completed_delivery":int(orders.loc[orders.status=="completed","delivery_minutes"].isna().sum()),
             "missing_cancelled_delivery":int(orders.loc[orders.status=="cancelled","delivery_minutes"].isna().sum()),
             "completed_revenue":round(orders.loc[orders.status=="completed","order_amount"].sum(),2),
             "city_counts":customers.city.value_counts().to_dict(),"gender_counts":customers.gender.value_counts().to_dict()}
    (OUT/"part_a_cleaning_stats.json").write_text(__import__('json').dumps(stats,indent=2),encoding='utf-8')
    validate(stats)
    return stats


def validate(s):
    assert s["customers"] == 1200 and s["products"] == 40 and s["orders"] == 12309
    assert s["completed"] == 11110 and s["cancelled"] == 1199
    assert s["missing_amount"] == 0 and s["missing_completed_delivery"] == 0 and s["missing_cancelled_delivery"] == 1199
    assert abs(s["completed_revenue"]-2998668.80)<0.01
    assert set(s["city_counts"]) == {"Pune","Delhi","Bengaluru","Chennai","Hyderabad","Mumbai"}


def run_sql():
    queries = [
    ("Q1", "SELECT customer_id, SUM(order_amount) AS spend_over_500 FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000 ORDER BY spend_over_500 DESC"),
    ("Q2", "SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue_inr FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category"),
    ("Q3", "SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC"),
    ("Q4", "SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id"),
    ("Q5", "WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.customer_id,spend.total_spend FROM spend,avg_spend WHERE spend.total_spend>avg_spend.avg_total ORDER BY spend.total_spend DESC"),
    ("Q6", "WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) row_num FROM spend ORDER BY city,rnk,customer_id"),
    ("Q7", "WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily ORDER BY day")]
    (ROOT/"sql/queries.sql").write_text("\n\n".join(f"-- {n}\n{q};" for n,q in queries)+"\n",encoding="utf-8")
    summaries={}
    with sqlite3.connect(DB) as conn:
        for name,q in queries:
            df=pd.read_sql_query(q,conn); df.to_csv(OUT/f"{name.lower()}_result.csv",index=False)
            summaries[name]={"rows":len(df)}
            if name=="Q2": summaries[name]["revenue_total"]=round(df.revenue_inr.sum(),2); summaries[name]["largest_cell"]=df.sort_values("revenue_inr",ascending=False).iloc[0].to_dict()
            if name=="Q5": summaries[name]["avg_spend"]=round(pd.read_sql_query("SELECT AVG(total) FROM (SELECT SUM(order_amount) total FROM orders WHERE status='completed' GROUP BY customer_id)",conn).iloc[0,0],2)
            if name=="Q6": summaries[name]["rank_row_number_different"]=int((df.rnk!=df.row_num).sum()); summaries[name]["city_top"]=df.sort_values(["city","rnk"]).groupby("city").first().reset_index()[["city","customer_id","total_spend"]].to_dict("records")
            if name=="Q7": summaries[name]["last_cumulative"]=round(df.cumulative_revenue.iloc[-1],2)
    clean_o=pd.read_csv(CLEAN/"orders.csv"); clean_c=pd.read_csv(CLEAN/"customers.csv"); clean_p=pd.read_csv(CLEAN/"products.csv")
    comp=clean_o[clean_o.status=="completed"].merge(clean_c[["customer_id","city"]],on="customer_id").merge(clean_p[["product_id","category"]],on="product_id")
    pivot=pd.pivot_table(comp,index="city",columns="category",values="order_amount",aggfunc="sum",fill_value=0,margins=True)
    pivot.to_csv(OUT/"city_category_pivot.csv")
    city=comp.groupby("city").order_amount.agg(sum="sum",mean="mean",count="count").round({"sum":2,"mean":2})
    city.to_csv(OUT/"city_spend_stats.csv")
    x=clean_o.loc[clean_o.status=="completed","order_amount"]
    q1,q3=x.quantile([.25,.75]); iqr=q3-q1; low=q1-1.5*iqr; high=q3+1.5*iqr
    corr=clean_o.loc[clean_o.status=="completed",["order_amount","discount_pct","quantity","delivery_minutes"]].corr()
    corr.to_csv(OUT/"completed_order_correlations.csv")
    pairs=corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().abs().sort_values(ascending=False)
    summaries["EDA"]={"mean":round(x.mean(),2),"median":round(x.median(),2),"skew":round(x.skew(),2),"q1":round(q1,2),"q3":round(q3,2),"lower_fence":round(low,2),"upper_fence":round(high,2),"outliers_below":int((x<low).sum()),"outliers_above":int((x>high).sum()),"largest_abs_correlation_pair":str(pairs.index[0]),"largest_abs_correlation":round(float(pairs.iloc[0]),3)}
    (OUT/"part_a_sql_summary.json").write_text(__import__('json').dumps(summaries,indent=2,default=str),encoding='utf-8')
    return summaries


def run():
    a=audit(); s=clean_and_store(); q=run_sql()
    print("Part A audit and SQL analysis complete.")
    return {"audit":a,"cleaning":s,"sql":q}

if __name__ == "__main__": run()
