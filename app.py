import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer

try:
    from missingpy import MissForest
    HAS_MISSFOREST = True
except Exception:
    HAS_MISSFOREST = False

try:
    import reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

st.set_page_config(page_title="AI‑Augmented Survey Processor", layout="wide")
st.title("📊 AI‑Augmented Survey Data Processing & Analysis")

if "state" not in st.session_state:
    st.session_state.state = {
        "df": None,
        "cleaned": None,
        "weight_col": None,
        "strata_col": None,
        "cluster_col": None,
        "logs": [],
        "rules": [],
        "anomaly_scores": None,
        "dp_epsilon": 0.0,
    }

state = st.session_state.state

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "1) Upload & Profile",
        "2) Cleaning",
        "3) Validation Rules",
        "4) Weights & Estimation",
        "5) Dashboards",
        "6) Reports",
        "7) Export"
    ])
    st.divider()
    st.caption("Tip: Use a column named 'weight' if you have design weights.")

# Utilities

def log(msg):
    state["logs"].append(msg)

def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def cat_cols(df):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

# 1) Upload & Profile
if page.startswith("1"):
    uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    schema_json = st.text_area("Optional: JSON schema mapping (rename, dtypes)", height=120,
                               placeholder='{"rename": {"old":"new"}, "dtypes": {"age":"Int64"}}')
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        if schema_json.strip():
            try:
                schema = json.loads(schema_json)
                if "rename" in schema:
                    df = df.rename(columns=schema["rename"]) 
                    log("Applied schema rename")
                if "dtypes" in schema:
                    for c, dt in schema["dtypes"].items():
                        try:
                            df[c] = df[c].astype(dt)
                        except Exception:
                            pass
                    log("Applied dtype coercions")
            except Exception as e:
                st.warning(f"Schema parse error: {e}")
        state["df"] = df.copy()
        st.success("File loaded")
        st.subheader("Preview")
        st.dataframe(df.head(20))
        st.subheader("Profiling")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Rows", len(df))
        with col2: st.metric("Columns", df.shape[1])
        with col3: st.metric("Missing cells", int(df.isna().sum().sum()))
        st.write("Missing by column")
        miss = df.isna().sum().sort_values(ascending=False)
        st.bar_chart(miss)
        st.write("Type breakdown")
        types = df.dtypes.astype(str).value_counts()
        st.bar_chart(types)

# 2) Cleaning
elif page.startswith("2"):
    if state["df"] is None:
        st.info("Upload data first")
    else:
        df = state["df"].copy()
        st.subheader("Missing‑Value Imputation")
        impute_method = st.selectbox("Method", ["None", "Mean", "Median", "Mode", "KNN", "Iterative", "MissForest" if HAS_MISSFOREST else "MissForest (install missingpy)"])
        if impute_method == "KNN":
            knn_k = st.slider("K for KNN", 2, 15, 5)
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Detector", ["None", "Z‑Score (3σ)", "IQR (1.5×IQR)", "Isolation Forest"])
        contam = 0.05
        if outlier_method == "Isolation Forest":
            contam = st.slider("Contamination (expected outlier fraction)", 0.01, 0.20, 0.05, 0.01)
        dp_epsilon = st.slider("Differential Privacy epsilon (0 = off)", 0.0, 5.0, state.get("dp_epsilon",0.0), 0.1)
        state["dp_epsilon"] = dp_epsilon
        run = st.button("Run Cleaning")
        if run:
            cleaned = df.copy()
            nums = numeric_cols(cleaned)
            if impute_method != "None":
                if impute_method == "Mean":
                    imputer = SimpleImputer(strategy="mean")
                elif impute_method == "Median":
                    imputer = SimpleImputer(strategy="median")
                elif impute_method == "Mode":
                    imputer = SimpleImputer(strategy="most_frequent")
                elif impute_method == "KNN":
                    imputer = KNNImputer(n_neighbors=knn_k)
                elif impute_method == "Iterative":
                    imputer = IterativeImputer()
                elif impute_method.startswith("MissForest"):
                    if HAS_MISSFOREST:
                        imputer = MissForest()
                    else:
                        imputer = SimpleImputer(strategy="median")
                try:
                    cleaned[nums] = imputer.fit_transform(cleaned[nums])
                    log(f"Imputation: {impute_method}")
                except Exception as e:
                    log(f"Imputation failed: {e}")
            if outlier_method != "None" and len(nums) > 0:
                if outlier_method.startswith("Z"):
                    z = (cleaned[nums] - cleaned[nums].mean())/cleaned[nums].std(ddof=0)
                    mask = (np.abs(z) < 3).all(axis=1)
                    cleaned = cleaned[mask]
                    log("Outliers removed via Z‑score 3σ")
                elif outlier_method.startswith("IQR"):
                    Q1 = cleaned[nums].quantile(0.25)
                    Q3 = cleaned[nums].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = ~(((cleaned[nums] < (Q1 - 1.5*IQR)) | (cleaned[nums] > (Q3 + 1.5*IQR))).any(axis=1))
                    cleaned = cleaned[mask]
                    log("Outliers removed via IQR 1.5×IQR")
                else:
                    try:
                        iso = IsolationForest(contamination=contam, random_state=42)
                        yhat = iso.fit_predict(cleaned[nums])
                        cleaned = cleaned[yhat == 1]
                        log("Outliers removed via Isolation Forest")
                    except Exception as e:
                        log(f"Isolation Forest failed: {e}")
            state["cleaned"] = cleaned
            st.success("Cleaning complete")
        if state["cleaned"] is not None:
            st.subheader("Cleaned Preview")
            st.dataframe(state["cleaned"].head(20))
            st.caption(f"Rows: {len(state['cleaned'])}  |  Columns: {state['cleaned'].shape[1]}")

# 3) Validation Rules
elif page.startswith("3"):
    if state["cleaned"] is None and state["df"] is not None:
        state["cleaned"] = state["df"].copy()
    if state["cleaned"] is None:
        st.info("Upload data first")
    else:
        st.subheader("Rule Builder")
        st.caption("Add simple constraints: required, min/max, regex, unique, allowed values")
        cols = list(state["cleaned"].columns)
        with st.form("rule_form"):
            col = st.selectbox("Column", cols)
            required = st.checkbox("Required (no NA)")
            minv = st.text_input("Min (numeric)")
            maxv = st.text_input("Max (numeric)")
            regex = st.text_input("Regex pattern (text)")
            unique = st.checkbox("Unique")
            allowed = st.text_input("Allowed values (comma‑separated)")
            add = st.form_submit_button("Add Rule")
        if add:
            rule = {"col": col, "required": required, "min": minv, "max": maxv, "regex": regex, "unique": unique, "allowed": [v.strip() for v in allowed.split(',')] if allowed else None}
            state["rules"].append(rule)
        if state["rules"]:
            st.write(pd.DataFrame(state["rules"]))
            if st.button("Run Validation"):
                df = state["cleaned"].copy()
                failures = []
                for r in state["rules"]:
                    c = r["col"]
                    if r.get("required"):
                        idx = df[df[c].isna()].index.tolist()
                        failures += [(c, i, "required") for i in idx]
                    try:
                        if r.get("min") not in (None,""):
                            mn = float(r["min"])
                            idx = df[df[c] < mn].index.tolist()
                            failures += [(c, i, f"min<{mn}") for i in idx]
                        if r.get("max") not in (None,""):
                            mx = float(r["max"])
                            idx = df[df[c] > mx].index.tolist()
                            failures += [(c, i, f"max>{mx}") for i in idx]
                    except Exception:
                        pass
                    if r.get("regex"):
                        pat = re.compile(r["regex"]) if r["regex"] else None
                        if pat is not None:
                            bad = df[~df[c].astype(str).str.match(pat, na=True)].index.tolist()
                            failures += [(c,i,"regex") for i in bad]
                    if r.get("unique"):
                        dup = df[df[c].duplicated(keep=False)].index.tolist()
                        failures += [(c,i,"unique") for i in dup]
                    if r.get("allowed"):
                        vals = set([v for v in r["allowed"] if v!=''])
                        if vals:
                            bad = df[~df[c].isin(vals)].index.tolist()
                            failures += [(c,i,"allowed") for i in bad]
                fail_df = pd.DataFrame(failures, columns=["column","row_index","rule"])
                if len(fail_df)==0:
                    st.success("No validation failures")
                else:
                    st.error(f"{len(fail_df)} validation failures")
                    st.dataframe(fail_df.head(200))
                    csv = fail_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="validation_failures.csv">Download failures</a>', unsafe_allow_html=True)

# 4) Weights & Estimation
elif page.startswith("4"):
    if state["cleaned"] is None:
        st.info("Clean the data first")
    else:
        df = state["cleaned"].copy()
        st.subheader("Design Columns")
        cols = df.columns.tolist()
        weight_col = st.selectbox("Weight column (optional)", [None]+cols, index=(cols.index("weight")+1 if "weight" in cols else 0))
        state["weight_col"] = weight_col
        st.subheader("Estimation")
        numcols = numeric_cols(df)
        tgt = st.selectbox("Target numeric variable", numcols)
        conf = st.slider("Confidence %", 80, 99, 95)
        group = st.multiselect("Group by (optional)", cat_cols(df))
        dp_eps = state.get("dp_epsilon", 0.0)
        if st.button("Compute Estimates"):
            out = []
            z = {80:1.282, 90:1.645, 95:1.960, 99:2.576}.get(conf, 1.960)
            if not group:
                g = [("ALL", df)]
            else:
                g = list(df.groupby(group))
            for key, sub in g:
                if weight_col:
                    w = sub[weight_col].astype(float).values
                    x = sub[tgt].astype(float).values
                    mu = np.average(x, weights=w)
                    var_w = np.average((x-mu)**2, weights=w)
                    neff = (w.sum()**2)/( (w**2).sum() ) if (w**2).sum() > 0 else len(sub)
                    se = np.sqrt(var_w/neff)
                else:
                    x = sub[tgt].astype(float).values
                    mu = x.mean()
                    se = x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0
                moe = z*se
                if dp_eps and dp_eps>0:
                    scale = moe/max(dp_eps,1e-6)
                    noise = np.random.laplace(0, scale)
                    mu_dp = mu + noise
                else:
                    mu_dp = mu
                out.append({"group": key if isinstance(key, str) else str(key), "mean": mu_dp, "moe": moe, "n": len(sub)})
            res = pd.DataFrame(out)
            st.dataframe(res)
            fig, ax = plt.subplots()
            ax.errorbar(range(len(res)), res["mean"], yerr=res["moe"], fmt="o")
            ax.set_xticks(range(len(res)))
            ax.set_xticklabels(res["group"], rotation=45, ha='right')
            ax.set_ylabel(f"Mean of {tgt}")
            st.pyplot(fig)
            state["estimates"] = res

# 5) Dashboards
elif page.startswith("5"):
    if state["cleaned"] is None:
        st.info("Run cleaning first")
    else:
        df = state["cleaned"].copy()
        st.subheader("Distributions")
        cols = numeric_cols(df)
        if cols:
            sel = st.selectbox("Column", cols)
            fig, ax = plt.subplots()
            sns.histplot(df[sel], kde=True, ax=ax)
            st.pyplot(fig)
        st.subheader("Correlations")
        if len(cols)>=2:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        st.subheader("Missingness")
        miss = df.isna().sum().sort_values(ascending=False)
        st.bar_chart(miss)

# 6) Reports
elif page.startswith("6"):
    st.subheader("Generate Report")
    fmt = st.selectbox("Format", ["HTML", "PDF" if HAS_REPORTLAB else "PDF (install reportlab)"])
    title = st.text_input("Report Title", "Survey Processing Report")
    author = st.text_input("Author", "Analyst")
    if st.button("Build Report"):
        df = state.get("cleaned") if state.get("cleaned") is not None else state.get("df")
        if df is None:
            st.warning("No data")
        else:
            buf = io.StringIO()
            buf.write(f"<h1>{title}</h1><p><i>{author}</i></p>")
            buf.write(f"<h2>Overview</h2><p>Rows: {len(df)} | Columns: {df.shape[1]}</p>")
            if state.get("estimates") is not None:
                buf.write("<h2>Estimates</h2>")
                buf.write(state["estimates"].to_html(index=False))
            buf.write("<h2>Logs</h2><pre>")
            for L in state["logs"]:
                buf.write(str(L)+"\n")
            buf.write("</pre>")
            html = buf.getvalue()
            b64 = base64.b64encode(html.encode()).decode()
            st.markdown(f'<a download="report.html" href="data:text/html;base64,{b64}">Download HTML</a>', unsafe_allow_html=True)
            if fmt.startswith("PDF") and HAS_REPORTLAB:
                pdf_path = "/tmp/report.pdf"
                c = canvas.Canvas(pdf_path, pagesize=A4)
                width, height = A4
                text = c.beginText(40, height-40)
                text.textLine(title)
                text.textLine(f"Author: {author}")
                text.textLine("")
                text.textLine(f"Rows: {len(df)} | Columns: {df.shape[1]}")
                text.textLine("")
                for L in state["logs"][:100]:
                    text.textLine(str(L))
                c.drawText(text)
                c.showPage(); c.save()
                with open(pdf_path,'rb') as f:
                    b64p = base64.b64encode(f.read()).decode()
                st.markdown(f'<a download="report.pdf" href="data:application/pdf;base64,{b64p}">Download PDF</a>', unsafe_allow_html=True)

# 7) Export
elif page.startswith("7"):
    st.subheader("Data & Metadata Export")
    if state.get("cleaned") is not None:
        csv = state["cleaned"].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="cleaned.csv">Download cleaned.csv</a>', unsafe_allow_html=True)
    if state.get("estimates") is not None:
        csv2 = state["estimates"].to_csv(index=False)
        b64 = base64.b64encode(csv2.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="estimates.csv">Download estimates.csv</a>', unsafe_allow_html=True)
    spec = {
        "provenance": state.get("logs", []),
        "rules": state.get("rules", []),
        "dp_epsilon": state.get("dp_epsilon", 0.0)
    }
    st.download_button("Download workflow.json", data=json.dumps(spec, indent=2), file_name="workflow.json", mime="application/json")
