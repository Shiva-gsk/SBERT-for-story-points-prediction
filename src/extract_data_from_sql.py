"""extract_data_from_sql_v3.py -- PS-6 TAWOS extraction (schema-checked).

Pulls every Story_Point-labelled issue with all fields PS-6 needs, applies the quality filters in
pandas (so each filter's impact is visible), and writes:
  data/raw/sbert_project6_issues.csv   filtered, ready for the pipeline
  data/reports/filter_funnel.csv           rows remaining after each filter (put in report)
  data/reports/project_summary.csv         per-project size, date range, SP stats
  data/reports/resolution_status_profile.csv   value counts, to decide any Resolution restriction

Env vars: MYSQL_PASSWORD, MYSQL_DATABASE (required); MYSQL_HOST, MYSQL_PORT, MYSQL_USER, PROJECT_NAME,
          SP_MIN_EXCLUSIVE (default 0), SP_MAX (default 100),
          ALLOWED_RESOLUTIONS (optional comma list, e.g. "Done,Fixed,Complete"; default: any non-empty)
"""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PASSWORD, DATABASE = os.getenv("MYSQL_PASSWORD"), os.getenv("MYSQL_DATABASE")
if not PASSWORD or not DATABASE:
    raise RuntimeError("Set MYSQL_PASSWORD and MYSQL_DATABASE.")
PROJECT_NAME = os.getenv("PROJECT_NAME")
SP_MIN = float(os.getenv("SP_MIN_EXCLUSIVE", "0"))
SP_MAX = float(os.getenv("SP_MAX", "100"))
ALLOWED = [s.strip() for s in os.getenv("ALLOWED_RESOLUTIONS", "").split(",") if s.strip()]
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
REP = Path("data/reports"); REP.mkdir(parents=True, exist_ok=True)   # diagnostics kept OUT of data/raw so loaders never glob them

engine = create_engine(URL.create("mysql+pymysql", username=os.getenv("MYSQL_USER", "root"), password=PASSWORD,
                                  host=os.getenv("MYSQL_HOST", "localhost"), port=int(os.getenv("MYSQL_PORT", "3306")),
                                  database=DATABASE, query={"charset": "utf8mb4"}),
                       pool_pre_ping=True, connect_args={"connect_timeout": 10})

QUERY = text("""
    SELECT i.ID AS Issue_ID, i.Jira_ID, i.Issue_Key, i.Project_ID,
           p.Name AS Project_Name, r.Name AS Repository_Name,
           i.Type AS Issue_Type, i.Status, i.Resolution,
           i.Creation_Date, i.Estimation_Date, i.Resolution_Date,
           i.Story_Point,
           i.Title_Changed_After_Estimation,
           i.Description_Changed_After_Estimation,
           i.Story_Point_Changed_After_Estimation,
           i.Title, i.Description_Text
    FROM Issue i
    JOIN Project p ON p.ID = i.Project_ID
    LEFT JOIN Repository r ON r.ID = p.Repository_ID
    WHERE i.Story_Point IS NOT NULL
      AND (:project_name IS NULL OR p.Name = :project_name)
""")

try:
    with engine.connect() as con:
        df = pd.read_sql_query(QUERY, con, params={"project_name": PROJECT_NAME})
finally:
    engine.dispose()

if df["Issue_ID"].duplicated().any():
    raise ValueError("Duplicate Issue_ID rows: check joins.")

# ---------- normalise types ----------
df["Creation_Date"] = pd.to_datetime(df["Creation_Date"], errors="coerce")
for c in ("Estimation_Date", "Resolution_Date"):
    df[c] = pd.to_datetime(df[c], errors="coerce")
df["Story_Point"] = pd.to_numeric(df["Story_Point"], errors="coerce")
flags = ["Title_Changed_After_Estimation", "Description_Changed_After_Estimation", "Story_Point_Changed_After_Estimation"]
null_flags = {f: int(df[f].isna().sum()) for f in flags}
for f in flags:
    df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).astype(int)   # NULL treated as "not changed"
df["Title"] = df["Title"].fillna("").astype(str).str.strip()
df["Description_Text"] = df["Description_Text"].fillna("").astype(str).str.strip()
df["Resolution"] = df["Resolution"].fillna("").astype(str).str.strip()
df["SBERT_Text"] = (df["Title"] + "\n\n" + df["Description_Text"]).str.strip()

# value profile BEFORE filtering, for choosing a Resolution rule
prof = (df.assign(Resolution=df.Resolution.replace("", "<empty/NULL>"))
          .groupby(["Resolution", "Status"]).size().reset_index(name="n").sort_values("n", ascending=False))
prof.to_csv(REP / "resolution_status_profile.csv", index=False)

# ---------- filters with funnel ----------
funnel = [("Story_Point IS NOT NULL (start)", len(df))]
def step(name, mask):
    global df
    df = df[mask]; funnel.append((name, len(df)))

step("Creation_Date valid", df.Creation_Date.notna())
step("Resolved (Resolution non-empty)" if not ALLOWED else f"Resolution in {ALLOWED}",
     df.Resolution.ne("") if not ALLOWED else df.Resolution.isin(ALLOWED))
step(f"{SP_MIN} < Story_Point <= {SP_MAX}", (df.Story_Point > SP_MIN) & (df.Story_Point <= SP_MAX))
step("Story_Point not changed after estimation", df.Story_Point_Changed_After_Estimation.eq(0))
step("Title not changed after estimation", df.Title_Changed_After_Estimation.eq(0))
step("Description not changed after estimation", df.Description_Changed_After_Estimation.eq(0))
step("Non-empty text (title or description)", df.SBERT_Text.ne(""))

F = pd.DataFrame(funnel, columns=["Step", "Rows_remaining"]); F["Dropped"] = -F.Rows_remaining.diff().fillna(0).astype(int)
F.to_csv(REP / "filter_funnel.csv", index=False)

# ---------- export (chronological order, ties broken by ID) ----------
df = df.sort_values(["Project_ID", "Creation_Date", "Issue_ID"], kind="stable")
df.to_csv(OUT / "sbert_project6_issues.csv", index=False, encoding="utf-8-sig")

summ = (df.groupby(["Project_ID", "Project_Name", "Repository_Name"], dropna=False)
          .agg(n=("Issue_ID", "size"), first=("Creation_Date", "min"), last=("Creation_Date", "max"),
               sp_median=("Story_Point", "median"), sp_mean=("Story_Point", "mean"),
               sp_unique=("Story_Point", "nunique")).reset_index().sort_values("n", ascending=False))
summ["test_n_20pct"] = (summ.n * 0.2).round().astype(int)
summ.to_csv(REP / "project_summary.csv", index=False)

print("NULL flag counts (treated as 0):", null_flags)
print("\nFilter funnel:\n", F.to_string(index=False))
print("\nResolution/Status profile (top 15, pre-filter):\n", prof.head(15).to_string(index=False))
print("\nProjects:\n", summ.to_string(index=False))
print(f"\nWrote {len(df)} rows to {(OUT / 'sbert_project6_issues.csv').resolve()}")