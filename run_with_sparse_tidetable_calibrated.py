"""
End-to-end evaluation pipeline with sparse tide-table calibration.
- Flow: /mnt/data/流水数据(9).xlsx
- Official tide: /mnt/data/潮位数据.xlsx
- Tide-table (sparse, ~4 pts/day): /mnt/data/潮汐表.xlsx

Outputs under: /mnt/data/run_with_sparse_tidetable_calibrated/
- per_station_predictions/station_<id>.csv
- summary_with_tidetable_calibration.csv
- overall_aggregates.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
import unicodedata

OUT = Path("/mnt/data/run_with_sparse_tidetable_calibrated")
(OUT/"per_station_predictions").mkdir(parents=True, exist_ok=True)

# ===== 在配置或顶部区域新增 =====
MIN_SEQ_LEN = 30          # 进入评估的最小有效行数（以前硬编码60）
MIN_TRAIN_INIT = 24       # strict_1step_* 中的 min_train
MERGE_TOL = "1h"          # 可改 "2h" 做对比

# -------------------- Utilities --------------------
def find_col(cols, keys):
    for k in keys:
        for c in cols:
            if k in str(c):
                return c
    raise KeyError(f"Column with keys {keys} not found.")

def hampel_mask(x, k=3, nsig=3.5):
    x = pd.Series(x)
    med = x.rolling(2*k+1, center=True, min_periods=1).median()
    mad = (x-med).abs().rolling(2*k+1, center=True, min_periods=1).median()*1.4826
    return ((x-med).abs() <= nsig*(mad+1e-8)).values

def resample_uv(df):
    df=df.set_index("datetime").sort_index()
    r=df.resample("1h").mean()
    r["u"]=r["u"].interpolate(method="time", limit=2)
    r["v"]=r["v"].interpolate(method="time", limit=2)
    return r.reset_index()

def angle(u,v): 
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0

def circ_diff_deg(a,b): 
    return ((a - b + 180.0) % 360.0) - 180.0

# Harmonic dictionaries
CONS = {"M2":12.4206, "S2":12.0, "K1":23.9345, "O1":25.8193, "SN":14.77*24.0}

def harmonic_mat(times, t0=None):
    t0 = t0 or times.min()
    th = (times - t0)/np.timedelta64(1,"h")
    cols = {}
    for name,T in CONS.items():
        w = 2*np.pi/T
        cols[f"{name}_cos"] = np.cos(w*th)
        cols[f"{name}_sin"] = np.sin(w*th)
    H = pd.DataFrame(cols, index=times.index if isinstance(times,pd.Series) else None)
    H["intercept"] = 1.0
    return H, t0

def deriv_mat(times, names, t0):
    th = (times - t0)/np.timedelta64(1,"h")
    cols = []
    for nm in names:
        base, kind = nm.split("_")
        T = CONS[base]
        w = 2*np.pi/T
        if kind=="cos":
            cols.append(-w*np.sin(w*th))
        elif kind=="sin":
            cols.append( w*np.cos(w*th))
        else:
            cols.append(np.zeros_like(th, dtype=float))
    return np.column_stack(cols)

def norm_col(s: str) -> str:
    """标准化列名：全角->半角，去空格，统一大小写"""
    s = unicodedata.normalize('NFKC', str(s)).strip()
    return s.lower()

# -------------------- Load datasets --------------------
flow = pd.read_excel("流水数据(9).xlsx", sheet_name=0)
tide_off = pd.read_excel("潮位数据.xlsx", sheet_name=0)
tide_tbl = pd.read_excel("潮汐表.xlsx", sheet_name=0)  # sparse, ~4 pts/day

# Columns
date_col_f = find_col(flow.columns, ["日期"])
time_col_f = find_col(flow.columns, ["时间"])
speed_col  = find_col(flow.columns, ["流速"])
dir_col    = find_col(flow.columns, ["流向"])
station_col= find_col(flow.columns, ["位置"])

date_col_off = find_col(tide_off.columns, ["日期"])
time_col_off = find_col(tide_off.columns, ["时间"])
eta_col_off  = find_col(tide_off.columns, ["潮高","潮位","cm","毫米","米"])

date_col_tbl = find_col(tide_tbl.columns, ["日期"])
# ------ 放宽“时间”列匹配 ------
TIME_KEYS = ["时间","潮时","潮时(hh:mm)","潮时（hh:mm）","潮时(hh:mm:ss)","潮时(小时:分钟)","日期时间","时刻"]
time_col_tbl = None
cols_norm_map = {c: norm_col(c) for c in tide_tbl.columns}

for c, cn in cols_norm_map.items():
    if any(k in cn for k in [norm_col(k0) for k0 in TIME_KEYS]):
        time_col_tbl = c
        break

if time_col_tbl is None:
    print("[DEBUG] 潮汐表列名:", list(tide_tbl.columns))
    raise KeyError("未找到时间列(候选: TIME_KEYS)。请确认潮汐表里是否存在独立时间或合并日期时间列。")

eta_col_tbl  = find_col(tide_tbl.columns, ["潮高","潮位","cm","毫米","米"])

# 若时间列其实是“日期时间”合并列，拆成日期+时间（供后面一致处理）
if norm_col(time_col_tbl).startswith("日期时间"):
    # 拆分：先转 datetime，再派生日期与时间字符串
    dt_combined = pd.to_datetime(tide_tbl[time_col_tbl], errors="coerce")
    tide_tbl[date_col_tbl] = dt_combined.dt.date.astype(str)
    tide_tbl[time_col_tbl] = dt_combined.dt.time.astype(str)

# -------------------- Parse flow (unify to-direction) --------------------
flow["datetime"] = pd.to_datetime(flow[date_col_f].astype(str)+" "+flow[time_col_f].astype(str), errors="coerce")
speed_mps = pd.to_numeric(flow[speed_col], errors="coerce")/60.0
theta_to = (pd.to_numeric(flow[dir_col], errors="coerce")+180.0)%360.0
th = np.deg2rad(theta_to)
flow["u"] = speed_mps*np.sin(th); flow["v"] = speed_mps*np.cos(th)
flow = flow.rename(columns={station_col:"station"})[["station","datetime","u","v"]]
flow["station"] = pd.to_numeric(flow["station"], errors="coerce").astype("Int64")
flow = flow.dropna().astype({"station":"int"}).sort_values(["station","datetime"])

# -------------------- Official tide to hourly & smooth --------------------
tide_dt_off = pd.to_datetime(tide_off[date_col_off].astype(str)+" "+tide_off[time_col_off].astype(str),
                             errors="coerce", utc=True).dt.tz_convert(None)
eta_off = pd.to_numeric(tide_off[eta_col_off], errors="coerce").astype(float)
off = pd.DataFrame({"datetime": tide_dt_off, "eta_off": eta_off}).dropna()
off = off.set_index("datetime").resample("1h").interpolate("time").reset_index()
off["eta_off_smooth"] = off["eta_off"].rolling(5, center=True, min_periods=1).mean()

# -------------------- Sparse tide-table (local) --------------------
tide_dt_tbl = pd.to_datetime(tide_tbl[date_col_tbl].astype(str)+" "+tide_tbl[time_col_tbl].astype(str),
                             errors="coerce", utc=True).dt.tz_convert(None)
eta_tbl = pd.to_numeric(tide_tbl[eta_col_tbl], errors="coerce").astype(float)
tbl = pd.DataFrame({"datetime": tide_dt_tbl, "eta_tbl": eta_tbl}).dropna().sort_values("datetime")

# -------------------- Harmonic backbone from official --------------------
H_off, t0 = harmonic_mat(off["datetime"])
ridge_off = Ridge(alpha=0.1, fit_intercept=False).fit(H_off.values, off["eta_off_smooth"].values)
off["eta_off_hat"] = ridge_off.predict(H_off.values)

# -------------------- Calibrate to sparse tide-table --------------------
H_tbl, _ = harmonic_mat(tbl["datetime"], t0=t0)
off_at_tbl = (H_tbl.values @ ridge_off.coef_)  # predicted official-harmonic on tbl times

# Design matrix: eta_tbl ≈ c + s*(off_at_tbl) + H_tbl @ delta
X_calib = np.column_stack([np.ones(len(tbl)), off_at_tbl, H_tbl.values])
# Penalize delta more heavily than c/s
scale_vec = np.array([1.0, 1.0] + [10.0]*H_tbl.shape[1])
X_scaled = X_calib / scale_vec
ridge_calib = Ridge(alpha=1.0, fit_intercept=False).fit(X_scaled, tbl["eta_tbl"].values)
coef = ridge_calib.coef_ / scale_vec
c = coef[0]; s = coef[1]; delta = coef[2:]

# -------------------- Build η_loc, dη_loc/dt on hourly grid --------------------
H_full, _ = harmonic_mat(off["datetime"], t0=t0)
eta_off_hat_full = off["eta_off_hat"].values
eta_delta_full = H_full.values @ delta
off["eta_loc"] = c + s*eta_off_hat_full + eta_delta_full

# Analytical derivative
names = [k for k in H_full.columns if k!="intercept"]
beta_series = pd.Series(ridge_off.coef_, index=H_full.columns)
beta_off_vec = beta_series[names].values
delta_vec = pd.Series(delta, index=["intercept"]+names)[names].values
D = deriv_mat(off["datetime"], names, t0)
d_off = D @ beta_off_vec
d_delta = D @ delta_vec
off["deta_loc"] = s*d_off + d_delta
off["deta_off"] = off["eta_off_smooth"].diff()/(off["datetime"].diff().dt.total_seconds())

# -------------------- Merge local tide into flow & build features --------------------
qc=[]
for st,g in flow.groupby("station"):
    mu=hampel_mask(g["u"]); mv=hampel_mask(g["v"])
    qc.append(g.loc[mu & mv])
flow_qc = pd.concat(qc).sort_values(["station","datetime"])

rs=[]
for st,g in flow_qc.groupby("station"):
    r=resample_uv(g[["datetime","u","v"]]); r["station"]=st; rs.append(r)
flow_rs = pd.concat(rs).sort_values(["station","datetime"])

off_use = off[["datetime","eta_loc","deta_loc","eta_off_smooth","deta_off"]].dropna()
data = pd.merge_asof(
    flow_rs.sort_values("datetime"),
    off_use.sort_values("datetime"),
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta(MERGE_TOL)
).dropna()

def build_features(df):
    df=df.sort_values("datetime").copy()
    df["eta"]=df["eta_loc"]; df["deta"]=df["deta_loc"]
    df["ad"]=df["deta"].abs()
    df["flood"]=(df["deta"]>0).astype(int); df["ebb"]=(df["deta"]<0).astype(int)
    # turn gate around deta zero-crossings (±90min)
    sgn=np.sign(df["deta"].values); idx=np.where(np.diff(sgn)!=0)[0]+1
    df["turn_gate"]=0
    if len(idx)>0:
        tchg=df["datetime"].iloc[idx].values; t_all=df["datetime"].values
        for i,t in enumerate(t_all):
            dt=np.min(np.abs((tchg-t).astype("timedelta64[m]").astype(float)))
            df.loc[df.index[i],"turn_gate"]=1 if dt<=90 else 0
    eps=1e-8
    df["eta_c"]=(df["eta"]-df["eta"].expanding(20).mean())/(df["eta"].expanding(20).std().fillna(1.0)+eps)
    df["ad_c"] =(df["ad"] -df["ad"].expanding(20).mean()) /(df["ad"].expanding(20).std().fillna(1.0)+eps)
    # delta features vs official
    df["eta_delta"]=df["eta"]-df["eta_off_smooth"]
    df["deta_delta"]=df["deta"]-df["deta_off"]
    # AR terms
    for k in [1,2,3]:
        df[f"u_l{k}"]=df["u"].shift(k); df[f"v_l{k}"]=df[f"v"].shift(k)
    df["u_rm3"]=df["u"].shift(1).rolling(3, min_periods=1).mean()
    df["v_rm3"]=df["v"].shift(1).rolling(3, min_periods=1).mean()
    return df

feat = pd.concat([build_features(g).assign(station=st) for st,g in data.groupby("station")]).sort_values(["station","datetime"])
stations = sorted([int(s) for s in feat["station"].unique() if 1<=int(s)<=13])

def add_neighbor_lag1(df_all, stations):
    rows=[]
    for st,g in df_all.groupby("station"):
        merged=g[["datetime"]].copy()
        nb=[s for s in [st-1, st+1] if s in stations]
        for s in nb:
            gn=df_all[df_all["station"]==s][["datetime","u","v"]].copy()
            gn["u_nb_l1"]=gn["u"].shift(1); gn["v_nb_l1"]=gn["v"].shift(1)
            gn=gn[["datetime","u_nb_l1","v_nb_l1"]]
            merged=pd.merge_asof(
                merged.sort_values("datetime"), 
                gn.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta(MERGE_TOL)
            )
        ucols=[c for c in merged.columns if c.startswith("u_nb_l1")]
        vcols=[c for c in merged.columns if c.startswith("v_nb_l1")]
        merged["u_nb_l1_mean"]=merged[ucols].mean(axis=1)
        merged["v_nb_l1_mean"]=merged[vcols].mean(axis=1)
        out=g.merge(merged[["datetime","u_nb_l1_mean","v_nb_l1_mean"]], on="datetime", how="left")
        rows.append(out)
    return pd.concat(rows).sort_values(["station","datetime"])

feat = add_neighbor_lag1(feat, stations)

# restrict to recent 10 days if needed
tmax = feat["datetime"].max(); tmin = tmax - pd.Timedelta(days=10)
feat = feat[(feat["datetime"]>=tmin)&(feat["datetime"]<=tmax)].copy()

# ---- Modeling ----
def harmonic_df(times, t0=None):
    t0 = t0 or times.min()
    th = (times - t0)/np.timedelta64(1,"h")
    feats = {}
    for name,T in CONS.items():
        w = 2*np.pi/T
        feats[f"{name}_cos"]=np.cos(w*th); feats[f"{name}_sin"]=np.sin(w*th)
    H = pd.DataFrame(feats, index=times.index if isinstance(times,pd.Series) else None)
    H["intercept"]=1.0
    return H, t0

def build_residual_X(df):
    base=pd.DataFrame(index=df.index)
    base["eta_c"]=df["eta_c"]; base["ad_c"]=df["ad_c"]; base["ad"]=df["ad"]
    base["eta_ad"]=df["eta_c"]*df["ad_c"]; base["eta2"]=df["eta_c"]**2; base["ad2"]=df["ad_c"]**2
    base["flood"]=df["flood"]; base["ebb"]=df["ebb"]; base["turn_gate"]=df["turn_gate"]
    base["eta_delta"]=df["eta_delta"]; base["deta_delta"]=df["deta_delta"]
    for k in [1,2,3]:
        base[f"u_l{k}"]=df[f"u_l{k}"]; base[f"v_l{k}"]=df[f"v_l{k}"]
    base["u_rm3"]=df["u_rm3"]; base["v_rm3"]=df["v_rm3"]
    base["u_nb_l1_mean"]=df["u_nb_l1_mean"]; base["v_nb_l1_mean"]=df["v_nb_l1_mean"]
    X=pd.DataFrame(index=df.index)
    for c in base.columns:
        X[f"F_{c}"]=base[c]*df["flood"]
        X[f"E_{c}"]=base[c]*df["ebb"]
    X["bias"]=1.0
    return X.fillna(0.0)

def strict_1step_predict_stride(st_df, alpha=0.1, alpha_res=0.05, weight_power=1.2,
                                deta_mask=1e-6, min_train=MIN_TRAIN_INIT, stride=12):
    st_df=st_df.sort_values("datetime").reset_index(drop=True)
    preds=[]; ru=rv=cu=cv=None; t0h=None; last_refit=-1
    for t in range(min_train, len(st_df)):
        train=st_df.iloc[:t].dropna(subset=["u","v"]).copy()
        test=st_df.iloc[[t]].copy()
        if len(train)<10: 
            continue
        if (ru is None) or (t-last_refit)>=stride:
            Xtr_h,t0h=harmonic_df(train["datetime"])
            A=train["ad"].values; a95=np.nanpercentile(A,95) if np.isfinite(A).any() else 1.0
            w=np.clip(A/(a95+1e-12),0,1.0)**weight_power; w=np.where(A<deta_mask,0.0,w)
            ru=Ridge(alpha=alpha, fit_intercept=False).fit(Xtr_h.values, train["u"].values, sample_weight=w)
            rv=Ridge(alpha=alpha, fit_intercept=False).fit(Xtr_h.values, train["v"].values, sample_weight=w)
            Ftr=build_residual_X(train)
            res_u=train["u"].values - ru.predict(Xtr_h.values)
            res_v=train["v"].values - rv.predict(Xtr_h.values)
            cu=Ridge(alpha=alpha_res, fit_intercept=False).fit(Ftr.values, res_u)
            cv=Ridge(alpha=alpha_res, fit_intercept=False).fit(Ftr.values, res_v)
            last_refit=t
        Xte_h,_=harmonic_df(test["datetime"], t0=t0h)
        pu=ru.predict(Xte_h.values); pv=rv.predict(Xte_h.values)
        Fte=build_residual_X(test)
        du=cu.predict(Fte.values); dv=cv.predict(Fte.values)
        out=test[["datetime","u","v","ad"]].copy()
        out["u_hat_dir"]=pu+du; out["v_hat_dir"]=pv+dv
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

def strict_1step_speed_baseline_stride(st_df, min_train=MIN_TRAIN_INIT, stride=12):
    st_df=st_df.sort_values("datetime").reset_index(drop=True)
    obs_sp=[]; hat_sp=[]; dts=[]; ru=rv=None; last_refit=-1
    for t in range(min_train, len(st_df)):
        train=st_df.iloc[:t].dropna(subset=["u","v"]).copy(); test=st_df.iloc[[t]].copy()
        if len(train)<10: 
            continue
        if (ru is None) or (t-last_refit)>=stride:
            Xtr=pd.DataFrame({"1":1.0,"eta":train["eta"].values,"deta":train["deta"].values,"eta_deta":(train["eta"]*train["deta"]).values})
            ru=Ridge(alpha=1e-6, fit_intercept=False).fit(Xtr.values, train["u"].values)
            rv=Ridge(alpha=1e-6, fit_intercept=False).fit(Xtr.values, train["v"].values)
            last_refit=t
        Xte=pd.DataFrame({"1":[1.0],"eta":test["eta"].values,"deta":test["deta"].values,"eta_deta":(test["eta"]*test["deta"]).values})
        uh=ru.predict(Xte.values)[0]; vh=rv.predict(Xte.values)[0]
        if pd.notna(test["u"].values[0]) and pd.notna(test["v"].values[0]):
            obs_sp.append(float(np.hypot(test["u"].values[0], test["v"].values[0])))
            hat_sp.append(float(np.hypot(uh, vh)))
            dts.append(test["datetime"].values[0])
    return np.array(obs_sp), np.array(hat_sp), pd.to_datetime(dts)

# ------------- Evaluate all stations -------------
sum_rows=[]
debug_rows=[]  # 新增：记录每站处理情况
for st in stations:
    st_df = feat[feat["station"]==st].copy()
    rec = {
        'station': st,
        'n_raw': len(st_df),
        'span_hours': (st_df["datetime"].max()-st_df["datetime"].min()).total_seconds()/3600.0,
        'status': 'ok',
        'n_pred': 0,
        'missing_u_pct': float(st_df["u"].isna().mean()*100),
        'missing_v_pct': float(st_df["v"].isna().mean()*100)
    }
    if len(st_df) < MIN_SEQ_LEN:
        # 只要 >= min_train 仍允许尝试预测
        if len(st_df) < MIN_TRAIN_INIT + 2:
            rec['status'] = f'skip_len<{MIN_TRAIN_INIT+2}'
            debug_rows.append(rec)
            continue
        rec['status'] = 'short_seq'  # 记录但继续
    pred = strict_1step_predict_stride(st_df, stride=12, min_train=MIN_TRAIN_INIT)
    if pred.empty:
        rec['status'] = 'no_pred'
        debug_rows.append(rec)
        continue
    rec['n_pred'] = len(pred)
    debug_rows.append(rec)
    pred["station"]=st
    pred.to_csv(OUT/"per_station_predictions"/f"station_{st}.csv", index=False, encoding="utf-8-sig")
    # Direction metrics
    th_obs = angle(pred["u"].values, pred["v"].values)
    th_hat = angle(pred["u_hat_dir"].values, pred["v_hat_dir"].values)
    derr = np.abs(circ_diff_deg(th_obs, th_hat))
    ad = pred["ad"].values; med = np.nanmedian(ad); mask = ad>=med
    w = np.clip(ad/(np.nanpercentile(ad,95)+1e-12),0,1.0)
    # Speed metrics
    sp_obs, sp_hat, dts_sp = strict_1step_speed_baseline_stride(st_df, stride=12)
    sum_rows.append({
        "station": int(st),
        "n_eval": int(len(pred)),
        "speed_MAE_mps": float(np.nanmean(np.abs(sp_obs-sp_hat))) if len(sp_obs) else np.nan,
        "speed_RMSE_mps": float(np.sqrt(np.nanmean((sp_obs-sp_hat)**2))) if len(sp_obs) else np.nan,
        "dir_MAE_deg_overall": float(np.nanmean(derr)),
        "dir_MAE_deg_|dη|>=median": float(np.nanmean(derr[mask])) if mask.any() else np.nan,
        "dir_MAE_deg_weighted(|dη|)": float(np.nansum(derr*w)/np.nansum(w)) if np.nansum(w)>0 else np.nan
    })

debug_df = pd.DataFrame(debug_rows)
debug_df.to_csv(OUT/"debug_station_status.csv", index=False, encoding="utf-8-sig")

if not sum_rows:
    print({"note":"No station produced predictions.",
           "debug_file": str(OUT/"debug_station_status.csv")})
    # 可快速查看各站原因
    print(debug_df)
else:
    summary = pd.DataFrame(sum_rows).sort_values("station")
    summary_path = OUT/"summary_with_tidetable_calibration.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if len(summary):
        agg = pd.DataFrame({
            "metric":[
                "speed_MAE_mps","speed_RMSE_mps",
                "dir_MAE_deg_overall","dir_MAE_deg_|dη|>=median","dir_MAE_deg_weighted(|dη|)"
            ],
            "mean":[
                summary["speed_MAE_mps"].mean(),
                summary["speed_RMSE_mps"].mean(),
                summary["dir_MAE_deg_overall"].mean(),
                summary["dir_MAE_deg_|dη|>=median"].mean(),
                summary["dir_MAE_deg_weighted(|dη|)"].mean()
            ],
            "median":[
                summary["speed_MAE_mps"].median(),
                summary["speed_RMSE_mps"].median(),
                summary["dir_MAE_deg_overall"].median(),
                summary["dir_MAE_deg_|dη|>=median"].median(),
                summary["dir_MAE_deg_weighted(|dη|)"].median()
            ]
        })
        agg.to_csv(OUT/"overall_aggregates.csv", index=False, encoding="utf-8-sig")
        print({"saved": str(summary_path), "agg": str(OUT/'overall_aggregates.csv')})
