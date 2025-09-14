import os
import numpy as np
import pandas as pd
import unicodedata
import re
import math
import glob  # 新增

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========= 配置 =========
TIDE_FILE = '潮汐表.xlsx'
# FLOW_FILES = ['流水数据(8).xlsx']  # 只看(8)
# 改为自动发现所有 Excel：流水数据(8).xlsx、流水数据(9).xlsx 等
FLOW_FILES = sorted([f for f in glob.glob('流水数据(*).xlsx') if os.path.isfile(f)])
SITES = list(range(1, 14))  # 站位 1~13
TIME_FREQ = '30min'         # 潮汐特征重采样频率（分析用）
# 新增：观测期内插值配置
USE_IN_SAMPLE_INTERP = False  # 先关闭；如需开启，也请把 KEEP_ONLY_FILLED=False
MAX_GAP_HOURS = 4            # 连续缺口超过该阈值不插值（保持NaN，避免跨大缺口）
MERGE_TOL_MIN = 20           # 潮汐-海流对齐容差
MIN_DAY_OF_MONTH = 21        # 只分析每月21日及以后的数据（老师建议）
N_SPLITS = 5                 # TSCV 折数（样本不足自动退化为留出法）
OUT_XLSX = '潮汐_海流_关系分析.xlsx'
DEBUG = True
# 新增：导出插值结果
DUMP_INTERP_FLOW = True
INTERP_EXPORT_FILE = f'输出_海流_插值_{TIME_FREQ}.xlsx'
# 新增：导出控制（仅保留有数值的行，避免整页空白）
KEEP_ONLY_FILLED = True
# 建议：不开插值时该开关无效；若要用插值训练，把它改为 False，避免把稀疏站位直接删空
# KEEP_ONLY_FILLED = False
# 新增：主轴与滞后配置
USE_AXIS_ROTATION = True
USE_LAG_SEARCH = True
LAG_MAX_MINUTES = 120        # 在 ±120 分钟内搜索相位滞后
DIR_MAE_SPEED_MIN = 0.05     # 方向误差只统计速度>该阈值的样本
# 新增：预测配置（指定一天；None=自动取潮汐数据的最后一天）
PREDICT_DAY: str | None = '2025-08-28'   # 改这里即可
# 新增：预测多日与训练截止日（含）
PREDICT_DAYS = ['2025-08-29', '2025-08-30']   # 需要预测的日期（可继续追加）
TRAIN_END_DAY: str | None = '2025-08-28'      # 训练样本截止到该日(含当日)；None=用预测日前一日
PREDICT_BOTH_COMPONENTS = True                # True=同时预测主轴(a)与横向(c)
# 新增：当某一分量训练样本为0时允许回退（a-only 或 c-only）
ALLOW_A_OR_C_FALLBACK = True
# 旧的单日开关依然保留，不再使用
# PREDICT_DAY: str | None = '2025-08-28'
PRED_OUT_XLSX_TMPL = '预测改_{}.xlsx'
# 新增：是否用插值后的流水做训练（预测与真值仍用原始观测）
TRAIN_WITH_INTERP = True
PREDICT_ON_GRID_IF_NO_OBS = True       # 目标日无原始观测→按潮汐网格整日输出
PREDICT_GRID_FREQ = TIME_FREQ
SITES: list[int] | None = None 
# 列名（尽量兼容）
DATE_COL_TIDE = '日期'
TIME_COL_TIDE_CANDS = ['潮时（hh:mm）', '潮时', '时间']
HEIGHT_COL_TIDE_CANDS = ['潮高(cm)', '潮高（cm）', '潮高']

DATE_COL_FLOW = '日期'
TIME_COL_FLOW = '时间'
DIR_CANDS = ['流向(°)', '流向', '方向(°)', '方向']
SPD_MIN = '流速(m/min)'
SPD_MS  = '流速(m/s)'
LOC_CANDS = ['位置名称', '站位', '位置', '站点', '编号', '地点']

# ...existing code (配置区靠近 DIR_MAE_SPEED_MIN)...
RIDGE_ALPHA = 3.0          # 先维持你之前更好的设置
USE_INTERACTIONS = False   # 回退：先关掉交互项
# 置为 None 表示不做“c弱相关就置零”的回退，保持 c 参与拟合（与你之前更好的一次一致）
MIN_C_ABS_CORR: float | None = None

# ========= 工具 =========
def log(msg):
    if DEBUG:
        print(f'[LOG] {msg}', flush=True)

def build_datetime(df, date_col, time_col, out_col='日期时间'):
    df = df.dropna(subset=[date_col, time_col]).copy()  # 避免 SettingWithCopyWarning
    df[out_col] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
    return df.dropna(subset=[out_col])

def _norm_col(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize('NFKC', s)  # 全角->半角
    s = s.replace('（', '(').replace('）', ')').replace('：', ':')
    s = s.replace('°', '')  # 去掉度符，便于匹配
    s = s.replace(' ', '').replace('\u3000', '')
    return s.lower()

def detect_col_by_keywords(df: pd.DataFrame, keywords: list[str]) -> str | None:
    # 归一化所有列名，支持模糊包含匹配
    cmap = {c: _norm_col(c) for c in df.columns}
    for orig, norm in cmap.items():
        for kw in keywords:
            if kw in norm:
                return orig
    return None

# 0=北、顺时针 的航海角度 <-> 数学坐标系
def to_uv(speed_ms, dir_deg):
    theta = np.deg2rad(90.0 - (dir_deg % 360.0))  # 转到x轴为0°
    u = speed_ms * np.cos(theta)
    v = speed_ms * np.sin(theta)
    return u, v

def from_uv(u, v):
    speed = np.sqrt(u*u + v*v)
    ang_math = np.rad2deg(np.arctan2(v, u))
    dir_deg = (90.0 - ang_math) % 360.0
    return speed, dir_deg

def circ_mean_deg(a_deg):
    # 空样本保护，避免 RuntimeWarning
    a = np.asarray(a_deg, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    a = np.deg2rad(a % 360.0)
    ang = np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a)))
    return (np.rad2deg(ang) + 360.0) % 360.0

def circ_mae_deg(y_true, y_pred):
    d = (y_pred - y_true + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(d)))

# 新增：速度阈值加权的方向MAE
def circ_mae_deg_masked(dir_true, dir_pred, speed_true, min_speed=0.05):
    dir_true = np.asarray(dir_true); dir_pred = np.asarray(dir_pred); speed_true = np.asarray(speed_true)
    mask = speed_true > float(min_speed)
    if mask.sum() == 0:
        return circ_mae_deg(dir_true, dir_pred)
    d = (dir_pred[mask] - dir_true[mask] + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(d)))

# 新增：主轴拟合与旋转
def fit_main_axis(u: np.ndarray, v: np.ndarray):
    uv = np.vstack([u, v]).T
    uv = uv[~np.isnan(uv).any(axis=1)]
    if len(uv) < 2:
        return 0.0, 0.0
    cov = np.cov(uv.T)
    w, vecs = np.linalg.eigh(cov)  # 小->大
    idx = np.argmax(w)
    vx, vy = vecs[0, idx], vecs[1, idx]
    theta = math.atan2(vy, vx)  # 与x轴夹角（弧度）
    var_ratio = float(w[idx] / max(w.sum(), 1e-9))
    return float(theta), var_ratio

def rotate_uv(u: np.ndarray, v: np.ndarray, theta_rad: float):
    ct, st = math.cos(theta_rad), math.sin(theta_rad)
    a = u * ct + v * st  # 沿主轴
    c = -u * st + v * ct # 横向
    return a, c

def math_deg_to_north_bearing(theta_deg: float):
    return (90.0 - theta_deg) % 360.0

def estimate_step_minutes(dt_series: pd.Series, default_min=30):
    d = dt_series.sort_values().diff().dropna().dt.total_seconds() / 60.0
    if len(d) == 0:
        return default_min
    try:
        return int(round(d.mode().iloc[0]))
    except Exception:
        return int(round(d.median()))

def best_lag_steps_for_corr(a_series: pd.Series, x_series: pd.Series, step_min: int, max_minutes: int):
    # 返回最优步数与相关
    max_steps = max(1, int(max_minutes // max(1, step_min)))
    best_step, best_corr = 0, -np.inf
    for s in range(-max_steps, max_steps + 1):
        xs = x_series.shift(s)
        df = pd.DataFrame({'a': a_series, 'x': xs}).dropna()
        if len(df) < 6:
            continue
        # 方差保护，避免 divide warning
        if float(df['a'].std(ddof=0)) < 1e-12 or float(df['x'].std(ddof=0)) < 1e-12:
            continue
        c = df['a'].corr(df['x'])
        if c is not None and np.isfinite(c) and abs(c) > best_corr:
            best_corr = abs(c); best_step = s
    return best_step, (best_corr if np.isfinite(c) else np.nan)

def detect_first(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def resample_tide_and_features(tide_df, dt_col='日期时间', h_col='潮高(cm)', freq='30min'):
    t = tide_df[[dt_col, h_col]].dropna().sort_values(dt_col).copy()
    rng = pd.date_range(t[dt_col].min().floor('D'), t[dt_col].max().ceil('D'), freq=freq)
    t = t.set_index(dt_col).reindex(rng).interpolate(method='time').rename_axis(dt_col).reset_index()
    t.columns = [dt_col, h_col]

    # dH/dt（cm/分钟）
    dt_minutes = (t[dt_col].diff().dt.total_seconds() / 60.0).bfill()
    dh = t[h_col].diff().fillna(0.0)
    t['dh_dt'] = (dh / dt_minutes).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    t['abs_dh_dt'] = np.abs(t['dh_dt'])

    # 基于导数过零的潮汐相位
    sign = np.sign(t['dh_dt']).fillna(0)
    turning = sign.shift(1).fillna(0) * sign <= 0
    turning_idx = np.where(turning)[0]
    cycle_id = np.zeros(len(t), dtype=int); cid = 0; last = 0
    for idx in turning_idx:
        cycle_id[last:idx] = cid; cid += 1; last = idx
    cycle_id[last:] = cid
    t['cycle_id'] = cycle_id
    t['cycle_pos'] = t.groupby('cycle_id').cumcount()
    cyc_len = t.groupby('cycle_id')['cycle_pos'].transform('max').replace(0, 1)
    t['phase01'] = (t['cycle_pos'] / cyc_len).clip(0, 1)
    t['phase_sin'] = np.sin(2*np.pi*t['phase01'])
    t['phase_cos'] = np.cos(2*np.pi*t['phase01'])

    cyc_min = t.groupby('cycle_id')[h_col].transform('min')
    cyc_max = t.groupby('cycle_id')[h_col].transform('max')
    t['range_cm'] = (cyc_max - cyc_min).fillna(0.0)

    hour = t[dt_col].dt.hour + t[dt_col].dt.minute/60.0
    t['hour_sin'] = np.sin(2*np.pi*hour/24.0)
    t['hour_cos'] = np.cos(2*np.pi*hour/24.0)
    t['is_flood'] = (t['dh_dt'] > 0).astype(int)
    return t

# ========= 数据读取与对齐 =========
def load_tide():
    tide = pd.read_excel(TIDE_FILE, engine='openpyxl')
    # 兼容时间/潮高列
    tc = detect_col_by_keywords(tide, ['潮时', '时间'])
    hc = detect_col_by_keywords(tide, ['潮高'])
    if (tc is None) or (hc is None):
        raise RuntimeError(f'潮汐表缺少 潮时/潮高 列，现有列: {list(tide.columns)}')
    tide = build_datetime(tide, DATE_COL_TIDE, tc, '日期时间')
    tide = tide.rename(columns={hc: '潮高(cm)'})
    tide_feat = resample_tide_and_features(tide[['日期时间','潮高(cm)']], dt_col='日期时间', h_col='潮高(cm)', freq=TIME_FREQ)
    return tide_feat

def load_flow():
    frames = []
    for f in FLOW_FILES:
        # 修复：存在性判断
        if not os.path.exists(f):
            log(f'[WARN] 文件不存在: {f}')
            continue
        df = pd.read_excel(f, engine='openpyxl')
        if df.empty:
            log(f'[WARN] 空表: {f}')
            continue

        # 必要的“日期/时间”列
        date_col = detect_col_by_keywords(df, ['日期'])
        time_col = detect_col_by_keywords(df, ['时间'])
        if date_col is None or time_col is None:
            log(f'[WARN] {f} 缺少 日期/时间 列，现有列: {list[df.columns]}')
            continue

        # 站位列（更鲁棒）
        loc_col = detect_col_by_keywords(df, ['位置名称','位置编号','站位','位置','站点','编号','地点'])
        if loc_col is None:
            log(f'[WARN] {f} 未找到站位列（位置名称/站位/位置编号/编号等），现有列: {list(df.columns)}')
            continue

        # 流向列（更鲁棒）
        dir_col = detect_col_by_keywords(df, ['流向','方向','方位'])
        if dir_col is None:
            log(f'[WARN] {f} 未找到流向列（流向/方向/方位），现有列: {list(df.columns)}')
            continue

        # 流速列（优先 m/s，其次 m/min）
        spd_ms_col = detect_col_by_keywords(df, ['流速(m/s)','速度(m/s)','流速ms'])
        spd_mmin_col = detect_col_by_keywords(df, ['流速(m/min)','速度(m/min)','流速m/min'])

        df = build_datetime(df, date_col, time_col, '日期时间')

        # 标准化站位为 1~13 的整数（从字符串提取数字）
        def to_site(v):
            m = re.search(r'(\d+)', str(v))
            return int(m.group(1)) if m else None

        out = pd.DataFrame({
            '日期时间': df['日期时间'],
            '站位': df[loc_col].apply(to_site),
            '流向(°)': pd.to_numeric(df[dir_col], errors='coerce')
        })

        if spd_ms_col is not None:
            out['流速(m/s)'] = pd.to_numeric(df[spd_ms_col], errors='coerce')
        elif spd_mmin_col is not None:
            out['流速(m/s)'] = pd.to_numeric(df[spd_mmin_col], errors='coerce') / 60.0
        else:
            log(f'[WARN] {f} 未找到流速列（m/s 或 m/min），现有列: {list(df.columns)}')
            continue

        out = out.dropna(subset=['日期时间', '站位', '流速(m/s)', '流向(°)'])
        # 仅在 SITES 非空时做过滤
        if SITES is not None:
            out = out[out['站位'].isin(SITES)]
        if out.empty:
            log(f'[WARN] {f} 过滤后为空')
            continue
        frames.append(out)

    if not frames:
        raise RuntimeError('未读取到任何海流观测')
    flow = pd.concat(frames, ignore_index=True)

    # 只保留每月21日及以后（可按需调整）
    flow['day'] = flow['日期时间'].dt.day
    flow = flow[flow['day'] >= MIN_DAY_OF_MONTH].drop(columns=['day'])
    # 新增：观测期内插值（与潮汐特征频率一致）
    if USE_IN_SAMPLE_INTERP:
        log(f'在观测期内做插值：freq={TIME_FREQ}, max_gap_hours={MAX_GAP_HOURS}')
        flow = interpolate_flow_uv(flow, TIME_FREQ, MAX_GAP_HOURS)
        # 导出插值结果（设置列宽与时间格式，避免Excel显示#######）
        if DUMP_INTERP_FLOW:
            try:
                with pd.ExcelWriter(INTERP_EXPORT_FILE, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm') as w:
                    # 总表
                    flow_sorted = flow.sort_values(['站位','日期时间'])
                    flow_sorted.to_excel(w, sheet_name='all', index=False)
                    ws = w.sheets['all']
                    ws.set_column('A:A', 19)   # 日期时间
                    ws.set_column('B:B', 6)    # 站位
                    ws.set_column('C:D', 10)   # is_obs / is_interp
                    ws.set_column('E:H', 14)   # u/v/速度/方向
                    # 分站位
                    for site, g in flow.groupby('站位'):
                        name = f'站位{int(site)}'
                        gi = g.sort_values('日期时间')
                        gi.to_excel(w, sheet_name=name, index=False)
                        ws = w.sheets[name]
                        ws.set_column('A:A', 19)
                        ws.set_column('B:B', 6)
                        ws.set_column('C:D', 10)
                        ws.set_column('E:H', 14)
                log(f'已导出插值后海流数据: {INTERP_EXPORT_FILE}')
            except Exception as e:
                log(f'[WARN] 导出插值结果失败: {e}')
    return flow

def load_flow_raw() -> pd.DataFrame:
    """
    读取海流原始观测（不插值、不重采样），统一到:
    ['日期时间','站位','流速(m/s)','流向(°)']
    仅保留每月 >=21 日。
    """
    frames = []
    for f in FLOW_FILES:
        if not os.path.exists(f):
            log(f'[WARN] 文件不存在: {f}')
            continue
        df = pd.read_excel(f, engine='openpyxl')
        if df.empty:
            log(f'[WARN] 空表: {f}')
            continue

        date_col = detect_col_by_keywords(df, ['日期'])
        time_col = detect_col_by_keywords(df, ['时间'])
        if date_col is None or time_col is None:
            log(f'[WARN] {f} 缺少 日期/时间 列，现有列: {list(df.columns)}')
            continue

        loc_col = detect_col_by_keywords(df, ['位置名称','位置编号','站位','位置','站点','编号','地点'])
        if loc_col is None:
            log(f'[WARN] {f} 未找到站位列，现有列: {list(df.columns)}')
            continue

        dir_col = detect_col_by_keywords(df, ['流向','方向','方位'])
        if dir_col is None:
            log(f'[WARN] {f} 未找到流向列，现有列: {list(df.columns)}')
            continue

        spd_ms_col = detect_col_by_keywords(df, ['流速(m/s)','速度(m/s)','流速ms'])
        spd_mmin_col = detect_col_by_keywords(df, ['流速(m/min)','速度(m/min)','流速m/min'])

        df = build_datetime(df, date_col, time_col, '日期时间')

        def to_site(v):
            m = re.search(r'(\d+)', str(v))
            return int(m.group(1)) if m else None

        out = pd.DataFrame({
            '日期时间': df['日期时间'],
            '站位': df[loc_col].apply(to_site),
            '流向(°)': pd.to_numeric(df[dir_col], errors='coerce')
        })

        if spd_ms_col is not None:
            out['流速(m/s)'] = pd.to_numeric(df[spd_ms_col], errors='coerce')
        elif spd_mmin_col is not None:
            out['流速(m/s)'] = pd.to_numeric(df[spd_mmin_col], errors='coerce') / 60.0
        else:
            log(f'[WARN] {f} 未找到流速列（m/s 或 m/min），现有列: {list(df.columns)}')
            continue

        out = out.dropna(subset=['日期时间','站位','流速(m/s)','流向(°)'])
        # 仅在 SITES 非空时做过滤
        if SITES is not None:
            out = out[out['站位'].isin(SITES)]
        if out.empty:
            continue
        frames.append(out)

    if not frames:
        raise RuntimeError('未读取到任何海流观测')
    flow = pd.concat(frames, ignore_index=True)
    flow['day'] = flow['日期时间'].dt.day
    flow = flow[flow['day'] >= MIN_DAY_OF_MONTH].drop(columns=['day'])
    return flow

# 新增：在观测期内对海流数据做 u/v 空间的时间插值
def interpolate_flow_uv(flow: pd.DataFrame, freq: str, max_gap_hours: int) -> pd.DataFrame:
    """
    输入：['日期时间','站位','流速(m/s)','流向(°)']
    修复：在(网格 ∪ 原始时刻)的联合索引上按时间插值，避免丢失不整点观测锚点。
    """
    freq_td = pd.Timedelta(freq)
    freq_min = int(freq_td.total_seconds() // 60) or 1
    max_steps = max(1, int(max_gap_hours * 60 / freq_min))
    out = []
    for site, g in flow.groupby('站位'):
        g = g.sort_values('日期时间').copy()
        u, v = to_uv(g['流速(m/s)'].values, g['流向(°)'].values)
        g['u'], g['v'] = u, v
        g = g.groupby('日期时间', as_index=False).mean(numeric_only=True)  # 去重但保留原始时刻

        # 网格 + 原始时刻 联合索引
        idx_grid = pd.date_range(g['日期时间'].min().floor('D'),
                                 g['日期时间'].max().ceil('D'), freq=freq)
        idx_full = idx_grid.union(g['日期时间'])
        gi = g.set_index('日期时间').reindex(idx_full).sort_index()

        gi['is_obs'] = gi.index.isin(g.set_index('日期时间').index)
        gi['u'] = gi['u'].interpolate(method='time', limit=max_steps, limit_direction='both')
        gi['v'] = gi['v'].interpolate(method='time', limit=max_steps, limit_direction='both')

        gi_grid = gi.loc[idx_grid]
        gi_grid['is_interp'] = (~gi_grid['is_obs']) & (gi_grid['u'].notna() | gi_grid['v'].notna())

        spd, direc = from_uv(gi_grid['u'].values, gi_grid['v'].values)
        gi_grid['流速(m/s)'] = spd
        gi_grid['流向(°)'] = direc
        gi_grid['站位'] = site

        res = gi_grid.reset_index().rename(columns={'index': '日期时间'})[
            ['日期时间','站位','is_obs','is_interp','u','v','流速(m/s)','流向(°)']
        ]
        if KEEP_ONLY_FILLED:
            res = res[(res['u'].notna()) | (res['v'].notna())]
        out.append(res)
    return pd.concat(out, ignore_index=True)

def align_merge(flow, tide_feat):
    # 合并（最近邻，限制容差）
    merged = pd.merge_asof(
        flow.sort_values('日期时间'),
        tide_feat.sort_values('日期时间'),
        on='日期时间', direction='nearest',
        tolerance=pd.Timedelta(minutes=MERGE_TOL_MIN)
    )
    merged = merged.dropna(subset=['潮高(cm)', 'dh_dt'])
    return merged

# 新增：预测指定日期
def predict_one_day(tide_feat: pd.DataFrame,
                    data_merged: pd.DataFrame,
                    target_date: str | None,
                    obs_flow: pd.DataFrame,
                    train_end_day: str | None = None) -> str:
    """
    使用指定日期作为预测日：
    - 训练：每站用 <= train_end_day(含) 的样本（若None，默认=预测日前一日）
    - 预测：仅在“原始观测的站位+时间点(obs_flow)”上输出
    - 对齐真实值：来自 obs_flow（保留 m/s 与 m/min）
    - 若 PREDICT_BOTH_COMPONENTS=True，则同时预测主轴分量a与横向分量c
    """
    if data_merged.empty or tide_feat.empty:
        raise RuntimeError('无可用于训练或预测的数据')

    target_day = tide_feat['日期时间'].dt.normalize().max() if target_date is None \
        else pd.to_datetime(str(target_date)).normalize()

    # 训练截止日
    if train_end_day is None:
        cut_day = (target_day - pd.Timedelta(days=1)).normalize()
    else:
        cut_day = pd.to_datetime(str(train_end_day)).normalize()

    base_features = ['phase_sin','phase_cos','hour_sin','hour_cos','range_cm','is_flood']
    step_min = estimate_step_minutes(tide_feat['日期时间'])
    preds, train_info = [], []

    tide_all = tide_feat.sort_values('日期时间').copy()
    if tide_all[tide_all['日期时间'].dt.normalize() == target_day].empty:
        raise RuntimeError(f'潮汐特征中找不到该日期: {target_day.date()}')

    for site in SITES:
        g_all = data_merged[data_merged['站位'] == site].sort_values('日期时间').copy()
        if g_all.empty:
            train_info.append({'站位': site, '训练样本数': 0, '使用模型': 'None',
                               '主轴角(°N顺时针)': np.nan, '最佳滞后(分钟)': np.nan,
                               '训练截止日': cut_day.date(), '备注': '训练期无样本'})
            continue

        # 训练样本：<= cut_day（含当日）
        g_tr = g_all[g_all['日期时间'].dt.normalize() <= cut_day].copy()
        if g_tr.empty:
            train_info.append({'站位': site, '训练样本数': 0, '使用模型': 'None',
                               '主轴角(°N顺时针)': np.nan, '最佳滞后(分钟)': np.nan,
                               '训练截止日': cut_day.date(), '备注': '训练集为空'})
            continue

        # 主轴
        theta_rad, var_ratio = fit_main_axis(g_tr['u'].values, g_tr['v'].values)
        axis_deg = math_deg_to_north_bearing(math.degrees(theta_rad))
        a_tr, c_tr = rotate_uv(g_tr['u'].values, g_tr['v'].values, theta_rad)

        # 滞后搜索：a 用 dh_dt；c 在 dh_dt 与 h 中择优
        step_min = estimate_step_minutes(tide_feat['日期时间'])
        best_steps_a, _ = best_lag_steps_for_corr(pd.Series(a_tr), g_tr['dh_dt'].reset_index(drop=True),
                                                  step_min, LAG_MAX_MINUTES)
        # c: 两种驱动候选
        steps_c_dhdt, corr1 = best_lag_steps_for_corr(pd.Series(c_tr), g_tr['dh_dt'].reset_index(drop=True),
                                                      step_min, LAG_MAX_MINUTES)
        steps_c_h, corr2 = best_lag_steps_for_corr(pd.Series(c_tr), g_tr['潮高(cm)'].reset_index(drop=True),
                                                   step_min, LAG_MAX_MINUTES)
        if np.nan_to_num(corr2, nan=-1) > np.nan_to_num(corr1, nan=-1):
            c_driver, best_steps_c = '潮高(cm)', steps_c_h
        else:
            c_driver, best_steps_c = 'dh_dt', steps_c_dhdt

        # 评估 c 与驱动的相关（训练期）
        if c_driver == 'dh_dt':
            dfc = pd.DataFrame({'c': pd.Series(c_tr), 'x': g_tr['dh_dt'].shift(best_steps_c)}).dropna()
        else:
            dfc = pd.DataFrame({'c': pd.Series(c_tr), 'x': g_tr['潮高(cm)'].shift(best_steps_c)}).dropna()
        c_abs_corr = float(dfc['c'].corr(dfc['x'])) if len(dfc) >= 10 else 0.0
        weak_c = not np.isfinite(c_abs_corr) or abs(c_abs_corr) < (MIN_C_ABS_CORR if MIN_C_ABS_CORR is not None else -np.inf)

        # 特征：为 a 与 c 分别构造驱动及其绝对值
        g_tr = g_tr.reset_index(drop=True)
        g_tr['a_drv'] = g_tr['dh_dt'].shift(best_steps_a)
        g_tr['a_drv_abs'] = g_tr['abs_dh_dt'].shift(best_steps_a)
        if c_driver == 'dh_dt':
            g_tr['c_drv'] = g_tr['dh_dt'].shift(best_steps_c)
            g_tr['c_drv_abs'] = g_tr['abs_dh_dt'].shift(best_steps_c)
        else:
            g_tr['c_drv'] = g_tr['潮高(cm)'].shift(best_steps_c)
            g_tr['c_drv_abs'] = g_tr['c_drv'].abs()

        feat_a = ['a_drv','a_drv_abs'] + base_features
        feat_c = ['c_drv','c_drv_abs'] + base_features

        Xa = g_tr[feat_a].copy()
        for c in feat_a:
            Xa[c] = pd.to_numeric(Xa[c], errors='coerce')

        Xc = g_tr[feat_c].copy()
        for c in feat_c:
            Xc[c] = pd.to_numeric(Xc[c], errors='coerce')

        keep_a = np.isfinite(Xa.values).all(axis=1) & np.isfinite(a_tr)
        keep_c = np.isfinite(Xc.values).all(axis=1) & np.isfinite(c_tr)
        Xa, ya = Xa.loc[keep_a], a_tr[keep_a]
        Xc, yc = Xc.loc[keep_c], c_tr[keep_c]
        n_a, n_c = int(len(Xa)), int(len(Xc))
        n_tr = int(min(n_a, n_c)) if (PREDICT_BOTH_COMPONENTS and not ALLOW_A_OR_C_FALLBACK) else int(max(n_a, n_c))
        if n_tr == 0:
            train_info.append({'站位': site, '训练样本数': 0, '使用模型': 'None',
                               '主轴角(°N顺时针)': axis_deg,
                               '最佳滞后(分钟)': f'a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)}',
                               '训练截止日': cut_day.date(), '备注': f'训练清洗后样本为0(a={n_a}, c={n_c})'})
            continue

        # 训练器
        def train_y(X, y):
            if len(X) >= 8:
                mdl = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=3.0, random_state=42))])
                mdl.fit(X.values, y); return mdl, 'Ridge', None
            elif len(X) >= 3:
                mdl = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
                mdl.fit(X.values, y); return mdl, 'Linear', None
            else:
                denom = (np.abs(X.iloc[:,0].values) + 1e-6)  # 第一列为驱动
                k = float(np.median(y / denom)); return None, f'k*drv(k={k:.4g})', k

        model_a, used_a, k_a = (train_y(Xa, ya) if n_a > 0 else (None, 'None', 0.0))
        if PREDICT_BOTH_COMPONENTS:
            if n_c > 0 and not weak_c:
                model_c, used_c, k_c = train_y(Xc, yc)
            else:
                model_c, used_c, k_c = None, 'zero', 0.0   # 回退：c=0
            used_model = f'a:{used_a}|c:{used_c}'
        else:
            model_c, k_c, used_c = None, 0.0, 'N/A'
            used_model = used_a

        # 预测时刻：原始观测优先，其次网格
        obs_day_site = obs_flow[(obs_flow['站位'] == site) &
                                (obs_flow['日期时间'].dt.normalize() == target_day)].copy()
        source = 'obs'
        if obs_day_site.empty and PREDICT_ON_GRID_IF_NO_OBS:
            times_df = tide_all[tide_all['日期时间'].dt.normalize() == target_day][['日期时间']].drop_duplicates()
            source = 'grid'
        elif obs_day_site.empty:
            train_info.append({'站位': site, '训练样本数': n_tr, '使用模型': used_model,
                               '主轴角(°N顺时针)': axis_deg,
                               '最佳滞后(分钟)': f'a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)}',
                               '训练截止日': cut_day.date(), '备注': '目标日无原始观测且未启用网格预测'})
            continue
        else:
            times_df = obs_day_site[['日期时间']].drop_duplicates()

        # 构造目标日特征
        t = tide_all.copy()
        t['a_drv'] = t['dh_dt'].shift(best_steps_a)
        t['a_drv_abs'] = t['abs_dh_dt'].shift(best_steps_a)
        if c_driver == 'dh_dt':
            t['c_drv'] = t['dh_dt'].shift(best_steps_c)
            t['c_drv_abs'] = t['abs_dh_dt'].shift(best_steps_c)
        else:
            t['c_drv'] = t['潮高(cm)'].shift(best_steps_c)
            t['c_drv_abs'] = t['c_drv'].abs()

        # 预测阶段也要生成交互特征，保证与 feat_a/feat_c 一致
        if USE_INTERACTIONS:
            t['a_drv_x_range'] = t['a_drv'] * t['range_cm']
            t['a_drv_x_flood'] = t['a_drv'] * t['is_flood']
            t['c_drv_x_range'] = t['c_drv'] * t['range_cm']
            t['c_drv_x_flood'] = t['c_drv'] * t['is_flood']

        tide_for_merge_a = t[['日期时间'] + feat_a].sort_values('日期时间')
        tide_for_merge_c = t[['日期时间'] + feat_c].sort_values('日期时间')

        Xa_te = pd.merge_asof(times_df.sort_values('日期时间'), tide_for_merge_a,
                              on='日期时间', direction='nearest',
                              tolerance=pd.Timedelta(minutes=MERGE_TOL_MIN))
        Xc_te = pd.merge_asof(times_df.sort_values('日期时间'), tide_for_merge_c,
                              on='日期时间', direction='nearest',
                              tolerance=pd.Timedelta(minutes=MERGE_TOL_MIN))

        for c in feat_a: Xa_te[c] = pd.to_numeric(Xa_te[c], errors='coerce')
        for c in feat_c: Xc_te[c] = pd.to_numeric(Xc_te[c], errors='coerce')

        # 分别判断 a/c 的可用性；若允许回退，则至少保留 a 或 c 有效的时刻
        keep_a_te = np.isfinite(Xa_te[feat_a].values).all(axis=1)
        keep_c_te = np.isfinite(Xc_te[feat_c].values).all(axis=1)
        keep_any = keep_a_te | (keep_c_te if PREDICT_BOTH_COMPONENTS else keep_a_te)
        Xa_te = Xa_te.loc[keep_any]; Xc_te = Xc_te.loc[keep_any]
        idx_time = Xa_te['日期时间'].values
        if len(Xa_te) == 0:
            train_info.append({'站位': site, '训练样本数': n_tr, '使用模型': used_model,
                               '主轴角(°N顺时针)': axis_deg,
                               '最佳滞后(分钟)': f'a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)}',
                               '训练截止日': cut_day.date(), '备注': f'预测时刻({source})在容差内无潮汐特征'})
            continue

        # 预测
        # a 预测（若无模型但有样本则用经验式；若训练样本为0则置0）
        if n_a > 0:
            if model_a is not None: a_pred = model_a.predict(Xa_te[feat_a].values)
            else:                    a_pred = Xa_te['a_drv'].values * float(k_a)
        else:
            a_pred = np.zeros(len(Xa_te), dtype=float)

        if PREDICT_BOTH_COMPONENTS:
            if n_c > 0:
                if model_c is not None: c_pred = model_c.predict(Xc_te[feat_c].values)
                else:                   c_pred = Xc_te['c_drv'].values * float(k_c)
        else:
            c_pred = np.zeros_like(a_pred)

        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        u_pred = a_pred * ct - c_pred * st
        v_pred = a_pred * st + c_pred * ct
        spd_pred_ms, dir_pred = from_uv(u_pred, v_pred)

        pred_df = pd.DataFrame({
            '日期时间': idx_time,
            '站位': site,
            'a_pred': a_pred,
            'c_pred': c_pred,
            'u_pred': u_pred,
            'v_pred': v_pred,
            'speed_pred(m/s)': spd_pred_ms,
            'speed_pred(m/min)': spd_pred_ms * 60.0,
            'dir_pred(°)': dir_pred,
            '主轴角(°N顺时针)': axis_deg,
            '最佳滞后(分钟)': f'a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)}',
            'c驱动': c_driver,
            '训练截止日': cut_day.date(),
            '预测时刻来源': source,
            '使用模型': used_model
        })
        preds.append(pred_df)
        log(f'[PRED] 站位[{site}] a样本{n_a} c样本{n_c} 预测点{len(Xa_te)} 滞后a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)} 模型:{used_model}')

        train_info.append({'站位': site, '训练样本数(a)': n_a, '训练样本数(c)': n_c, '使用模型': used_model,
                           '主轴角(°N顺时针)': axis_deg,
                           '最佳滞后(分钟)': f'a:{int(best_steps_a*step_min)}/c:{int(best_steps_c*step_min)}',
                           '训练截止日': cut_day.date(),
        })

    if not preds:
        raise RuntimeError('没有任何站位可生成预测')

    out_pred = pd.concat(preds, ignore_index=True).sort_values(['站位','日期时间'])

    # 真实值（原始观测）
    obs_day = obs_flow[obs_flow['日期时间'].dt.normalize() == target_day][
        ['日期时间','站位','流速(m/s)','流向(°)']].copy()
    if not obs_day.empty:
        u_true, v_true = to_uv(obs_day['流速(m/s)'].values, obs_day['流向(°)'].values)
        obs_day['u_true'] = u_true
        obs_day['v_true'] = v_true
        obs_day = obs_day.rename(columns={'流向(°)':'dir_true(°)', '流速(m/s)':'speed_true(m/s)'})
        obs_day['speed_true(m/min)'] = obs_day['speed_true(m/s)'] * 60.0
        merged_check = pd.merge(out_pred, obs_day, on=['日期时间','站位'], how='left')
        # 误差
        if {'speed_true(m/s)', 'dir_true(°)'}.issubset(merged_check.columns):
            merged_check['abs_err_speed(m/s)']  = (merged_check['speed_pred(m/s)'] - merged_check['speed_true(m/s)']).abs()
            merged_check['abs_err_speed(m/min)'] = merged_check['abs_err_speed(m/s)'] * 60.0
            d = (merged_check['dir_pred(°)'] - merged_check['dir_true(°)'] + 180.0) % 360.0 - 180.0
            merged_check['abs_err_dir(°)'] = d.abs()
    else:
        merged_check = out_pred.copy()

    date_str = str(target_day.date())
    out_file = PRED_OUT_XLSX_TMPL.format(date_str)

    # 计算指标（含真值时）
    if {'speed_true(m/s)', 'dir_true(°)'}.issubset(merged_check.columns):
        eva = merged_check[merged_check['speed_true(m/s)'].notna()].copy()
        # 方向误差仅统计真值速度>阈值（减少驻水点的方向发散）
        mask_dir = eva['speed_true(m/s)'] > float(DIR_MAE_SPEED_MIN)
        eva.loc[~mask_dir, 'abs_err_dir(°)'] = np.nan
        metrics_by_site = eva.groupby('站位', dropna=False).agg(
            样本数=('speed_true(m/s)', 'size'),
            速度MAE_mps=('abs_err_speed(m/s)', 'mean'),
            速度RMSE_mps=('abs_err_speed(m/s)', lambda s: np.sqrt((s**2).mean())),
            方向MAE_deg=('abs_err_dir(°)', 'mean')
        ).reset_index()
        metrics_overall = pd.DataFrame([{
            '样本数': int(len(eva)),
            '速度MAE_mps': float(eva['abs_err_speed(m/s)'].mean()),
            '速度RMSE_mps': float(np.sqrt((eva['abs_err_speed(m/s)']**2).mean())),
            '方向MAE_deg': float(eva['abs_err_dir(°)'].mean(skipna=True)),
        }])
        # 终端打印简要
        log(f"[METRICS {date_str}] 速度MAE={metrics_overall.at[0,'速度MAE_mps']:.4f} m/s, "
            f"方向MAE(>{DIR_MAE_SPEED_MIN} m/s)={metrics_overall.at[0,'方向MAE_deg']:.1f}°")
    else:
        metrics_by_site = pd.DataFrame(columns=['站位','样本数','速度MAE_mps','速度RMSE_mps','方向MAE_deg'])
        metrics_overall = pd.DataFrame([{'样本数':0,'速度MAE_mps':None,'速度RMSE_mps':None,'方向MAE_deg':None}])

    with pd.ExcelWriter(out_file, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm') as w:
        merged_check.to_excel(w, sheet_name='predictions', index=False)
        ws = w.sheets['predictions']
        ws.set_column('A:A', 19)
        ws.set_column('B:B', 6)
        ws.set_column('C:Z', 14)
        pd.DataFrame(train_info).to_excel(w, sheet_name='train_info', index=False)
        metrics_by_site.to_excel(w, sheet_name='metrics_by_site', index=False)
        metrics_overall.to_excel(w, sheet_name='metrics_overall', index=False)

    log(f'✅ 已输出指定日期({date_str})预测: {out_file}')
    return out_file

# ========= 主分析 =========
def analyze():
    log('读取潮汐与海流数据...')
    tide_feat = load_tide()
    flow_raw = load_flow_raw()

    # 动态确定站位集合
    global SITES
    if (SITES is None) or (len(SITES) == 0):
        SITES = sorted(pd.Series(flow_raw['站位']).dropna().astype(int).unique().tolist())
        log(f'自动识别站位: {SITES}')

    # 训练数据源
    flow_train = flow_raw
    if TRAIN_WITH_INTERP:
        log(f'训练使用插值序列：freq={TIME_FREQ}, max_gap_hours={MAX_GAP_HOURS}')
        flow_train = interpolate_flow_uv(flow_raw, TIME_FREQ, MAX_GAP_HOURS)
        if DEBUG:
            cnt = flow_train.groupby('站位', dropna=False)['日期时间'].size().reset_index(name='样本数')
            log(f'插值后各站样本数(前若干)：{cnt.head(20).to_dict(orient="records")}')

    data = align_merge(flow_train, tide_feat)
    if data.empty:
        raise RuntimeError('潮汐-海流合并为空，请检查时间对齐与容差')

    u, v = to_uv(data[SPD_MS].values, data['流向(°)'].values)
    data['u'] = u; data['v'] = v
    data['speed'] = np.sqrt(u*u + v*v)

    outs = []
    for day in PREDICT_DAYS:
        out = predict_one_day(tide_feat, data, day, flow_raw, train_end_day=TRAIN_END_DAY)
        outs.append(out)
    log(f'✅ 完成预测: {", ".join(outs)}')

if __name__ == '__main__':
    analyze()