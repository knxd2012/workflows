import pandas as pd
import numpy as np
import lightgbm as lgb
import math
import os
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================================================================
# V79 實盤雙擎快取系統 (最終黃金結案版：動態型態生成 + 完整報表排版)
# =============================================================================================

MODEL_DIR = "models_v79"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "AH_V79_DUAL_T13H.pkl")

# 鎖定實盤黃金策略門檻
POLICY = {
    "hours_before_ko": 13.0,
    "prob_th": 0.53,       # 引擎 A：機率門檻
    "diff_th": 0.15,       # 引擎 B：淨勝球安全邊際
}

def parse_hc(hc_str):
    if pd.isna(hc_str) or str(hc_str).strip() == '': return 0.0
    orig = str(hc_str); is_a = '受让' in orig or '受讓' in orig
    c = orig.replace(' ', '').replace('受让', '').replace('受讓', '')
    m = {
        '平手': 0.0, '平手/半球': 0.25, '半球': 0.5, '半球/一球': 0.75, '一球': 1.0,
        '一球/球半': 1.25, '球半': 1.5, '球半/两球': 1.75, '两球': 2.0,
        '两球/两球半': 2.25, '两球半': 2.5, '两球半/三球': 2.75, '三球': 3.0,
        '三球/三球半': 3.25, '三球半': 3.5, '三球半/四球': 3.75, '四球': 4.0,
        '四球/四球半': 4.25, '四球半': 4.5, '四球半/五球': 4.75, '五球': 5.0
    }
    try: return -float(c) if is_a else float(c)
    except: return -m.get(c, 0.0) if is_a else m.get(c, 0.0)

# ============================================================
# 動態特徵生成：莊家盤口型態 (Pat_L, Pat_M)
# ============================================================
def generate_patterns(df_feat):
    req_cols = ['home_open_n1', 'home_open_n2', 'home_open_n3', 'home_open_n4',
                'away_open_n1', 'away_open_n2', 'away_open_n3', 'away_open_n4']
    
    missing = [c for c in req_cols if c not in df_feat.columns]
    if missing:
        if 'Pat_L' not in df_feat.columns: df_feat['Pat_L'] = '缺失'
        if 'Pat_M' not in df_feat.columns: df_feat['Pat_M'] = '缺失'
        return df_feat

    for c in req_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors='coerce')
        
    df_feat['low_side_open'] = np.where(df_feat['home_open_n1'] < df_feat['away_open_n1'], 'Home', 'Away')
    
    a1 = np.where(df_feat['low_side_open'] == 'Home', df_feat['home_open_n1'], df_feat['away_open_n1'])
    a2 = np.where(df_feat['low_side_open'] == 'Home', df_feat['home_open_n2'], df_feat['away_open_n2'])
    a3 = np.where(df_feat['low_side_open'] == 'Home', df_feat['home_open_n3'], df_feat['away_open_n3'])
    a4 = np.where(df_feat['low_side_open'] == 'Home', df_feat['home_open_n4'], df_feat['away_open_n4'])
    
    b1 = np.where(df_feat['low_side_open'] == 'Home', df_feat['away_open_n1'], df_feat['home_open_n1'])
    b2 = np.where(df_feat['low_side_open'] == 'Home', df_feat['away_open_n2'], df_feat['home_open_n2'])
    b3 = np.where(df_feat['low_side_open'] == 'Home', df_feat['away_open_n3'], df_feat['home_open_n3'])
    b4 = np.where(df_feat['low_side_open'] == 'Home', df_feat['away_open_n4'], df_feat['home_open_n4'])

    l1 = np.abs((b4 - a3) * 100).round()
    l2 = np.abs((a3 - b2) * 100).round()
    l3 = np.abs((b2 - a1) * 100).round()

    df_feat['Pat_L'] = np.where((l1 <= l2) & (l2 <= l3), '反階梯',
                         np.where((l1 >= l2) & (l2 >= l3), '順階梯',
                         np.where((l2 > l1) & (l2 > l3), '雙峰擠壓',
                         np.where((l2 < l1) & (l2 < l3), '凹陷誘盤', '常規波浪'))))

    m1 = np.abs((a4 - b3) * 100).round()
    m2 = np.abs((b3 - a2) * 100).round()
    m3 = np.abs((a2 - b1) * 100).round()

    df_feat['Pat_M'] = np.where((m1 > m2) & (m2 > m3), 'A>B>C',
                         np.where((m1 < m2) & (m2 < m3), 'A<B<C',
                         np.where((m1 > m2) & (m2 < m3), 'A>B<C',
                         np.where((m1 < m2) & (m2 > m3), 'A<B>C', '無序'))))
                         
    return df_feat

# ============================================================
# 核心資料清洗與 T-13H 對齊
# ============================================================
def prepare_dataset(raw_file, lines_file, is_train=True, verbose=True):
    if verbose: print(f"\n{'='*60}\n>> 清洗資料 {'[訓練模式]' if is_train else '[推論模式]'} : {raw_file}\n{'='*60}")
    
    if not os.path.exists(raw_file) or not os.path.exists(lines_file):
        return pd.DataFrame()

    raw_df = pd.read_csv(raw_file)
    raw_df['ko_dt'] = pd.to_datetime(raw_df['kickoff_time'], errors='coerce')
    raw_df = raw_df.dropna(subset=['ko_dt']).sort_values('ko_dt').drop_duplicates('match_id', keep='first').reset_index(drop=True)
    
    lines = pd.read_csv(lines_file)
    for col in ['home', 'away']: lines[col] = pd.to_numeric(lines[col], errors='coerce')
    lines = lines.merge(raw_df[['match_id', 'ko_dt']], on='match_id', how='inner')
    lines['line_dt'] = pd.to_datetime(lines['ko_dt'].dt.year.astype(str)+'-'+lines['time'].astype(str), format='%Y-%m-%d %H:%M', errors='coerce')
    lines.loc[lines['line_dt'] > lines['ko_dt'] + pd.Timedelta(days=1), 'line_dt'] -= pd.DateOffset(years=1)
    
    # 提煉初盤 (Opening Odds) 以便動態生成 Pat_L / Pat_M
    if 'home_open_n1' not in raw_df.columns:
        if verbose: print("   提取初盤軌跡以動態生成 Pat_L / Pat_M...")
        # 嚴格按時間排序後取 first()，保證抓到最古老的初盤
        open_lines = lines.sort_values(['match_id', 'line_dt']).groupby(['match_id', 'line']).first().unstack('line').reset_index()
        open_lines.columns = [f'{c[0]}_open_{c[1]}' if c[1] else c[0] for c in open_lines.columns]
        raw_df = raw_df.merge(open_lines, on='match_id', how='left')
    
    raw_df = generate_patterns(raw_df)
    
    # 提取 T-13H 切片
    lines['cutoff_dt'] = lines['ko_dt'] - pd.Timedelta(hours=POLICY["hours_before_ko"])
    t13_lines = lines[lines['line_dt'] <= lines['cutoff_dt']] \
        .sort_values(['match_id','line_dt']) \
        .groupby(['match_id','line']).last().reset_index()

    # 保留 line_dt
    t13_lines = t13_lines[['match_id','line','home','away','handicap','line_dt']]
    t13_lines = t13_lines.pivot(index='match_id', columns='line')
    t13_lines.columns = [f"{c[0]}_{c[1]}_T13H" for c in t13_lines.columns]
    t13_lines = t13_lines.reset_index()

    merged = raw_df.merge(t13_lines, on='match_id', how='inner')

    
    merged['bet_hc_num'] = merged['handicap_n1_T13H'].astype(str).apply(parse_hc)
    merged['bet_fav_is_home'] = np.where(merged['bet_hc_num'] >= 0, 1, 0)
    merged['bet_fav_hc'] = merged['bet_hc_num'].abs()
    
    merged['real_odds_fav'] = np.where(merged['bet_fav_is_home'] == 1, merged['home_n1_T13H'], merged['away_n1_T13H'])
    merged['real_odds_dog'] = np.where(merged['bet_fav_is_home'] == 1, merged['away_n1_T13H'], merged['home_n1_T13H'])

    if is_train:
        merged['g_fav'] = np.where(merged['bet_fav_is_home'] == 1, merged['home_goal'], merged['away_goal'])
        merged['g_dog'] = np.where(merged['bet_fav_is_home'] == 1, merged['away_goal'], merged['home_goal'])
        merged['net_diff'] = merged['g_fav'] - merged['g_dog'] - merged['bet_fav_hc']
        merged['Target_AH'] = np.where(merged['net_diff'] > 0, 1, np.where(merged['net_diff'] < 0, 0, np.nan))
        merged = merged.dropna(subset=['Target_AH', 'net_diff']).copy()
    
    cat_features = []
    for col in ['Pat_L', 'Pat_M', 'cross_pattern']:
        if col in merged.columns:
            merged[col] = merged[col].astype('category')
            cat_features.append(col)

    if verbose: print(f"✅ 成功對齊 T-{POLICY['hours_before_ko']}H 賽事：{len(merged)} 場")
    return merged

# ============================================================
# 訓練與封裝 PKL
# ============================================================
def train_and_save(train_df, force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"\n📂 載入現有雙擎模型：{MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle

    print(f"\n🔧 開始重新訓練 T-{POLICY['hours_before_ko']}H 雙擎模型...")
    
    leakage_cols = ['home_goal', 'away_goal', 'g_fav', 'g_dog', 'total_goals', 'result', 'halftime_score', 'Target_AH', 'net_diff', 'fav_win_match']
    meta_cols = ['match_id', 'kickoff_time', 'league', 'ko_dt', 'home_team', 'away_team']
    drop_cols = leakage_cols + meta_cols
    
    features = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype in [np.float64, np.int64, 'category']]
    cat_features = [c for c in features if train_df[c].dtype == 'category']

    meds = {}
    for c in features:
        if c not in cat_features:
            med = float(train_df[c].median())
            meds[c] = med
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(med)

    print(f"   特徵數：{len(features)} | 訓練場次：{len(train_df)}")

    clf_train = lgb.Dataset(train_df[features], label=train_df['Target_AH'], categorical_feature=cat_features)
    clf_model = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 4, 'verbose': -1}, clf_train, num_boost_round=350)
    
    reg_train = lgb.Dataset(train_df[features], label=train_df['net_diff'], categorical_feature=cat_features)
    reg_model = lgb.train({'objective': 'huber', 'metric': 'rmse', 'learning_rate': 0.03, 'max_depth': 4, 'verbose': -1}, reg_train, num_boost_round=450)

    bundle = {
        "clf_model": clf_model, "reg_model": reg_model,
        "features": features, "cat_features": cat_features,
        "meds": meds, "policy": POLICY,
        "version": "V79_FINAL_T13H", "train_date": str(datetime.now().date()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"💾 模型已成功儲存至：{MODEL_PATH}")
    return bundle

# ============================================================
# 每日實盤推論與報表輸出
# ============================================================
def predict_and_print(bundle, future_df, out_csv="ah_recommend_v79.csv"):
    if future_df.empty: return
    
    clf_model, reg_model = bundle["clf_model"], bundle["reg_model"]
    features, meds, pol = bundle["features"], bundle["meds"], bundle["policy"]

    df_inf = future_df.copy()
    
    for c in features:
        if c not in df_inf.columns:
            df_inf[c] = meds.get(c, 0.0) if c not in bundle["cat_features"] else np.nan
        if c not in bundle["cat_features"]:
            df_inf[c] = pd.to_numeric(df_inf[c], errors="coerce").fillna(meds.get(c, 0.0))
        else:
            df_inf[c] = df_inf[c].astype('category')

    df_inf['prob_Fav'] = clf_model.predict(df_inf[features])
    df_inf['prob_Dog'] = 1.0 - df_inf['prob_Fav']
    df_inf['pred_diff'] = reg_model.predict(df_inf[features])

    cond_fav = (df_inf['prob_Fav'] > pol['prob_th']) & (df_inf['pred_diff'] > pol['diff_th'])
    cond_dog = (df_inf['prob_Dog'] > pol['prob_th']) & (df_inf['pred_diff'] < -pol['diff_th'])

    df_inf['Action'] = np.where(cond_fav, 'FAV', np.where(cond_dog, 'DOG', 'SKIP'))
    rec = df_inf[df_inf["Action"] != "SKIP"].copy().sort_values("ko_dt")
    
    if len(rec) == 0:
        print("\n⚠️ 系統警報：今日無賽事觸發 T-13H 黃金門檻。")
        return pd.DataFrame()

    print(f"\n{'='*120}")
    print(f" 🎯 AH V79 最終實盤推薦清單   共篩出 {len(rec)} 場")
    print(f" 策略：T-{pol['hours_before_ko']}H | 雙擎共振 (機率 > {pol['prob_th']}, 淨勝預期 > {pol['diff_th']})")
    print(f"{'='*120}")

    cur_date = None
    for _, r in rec.iterrows():
        ko_dt = pd.to_datetime(r["ko_dt"])
        date_s, time_s = ko_dt.strftime("%Y-%m-%d"), ko_dt.strftime("%H:%M")

        if date_s != cur_date:
            print(f"\n 📅 {date_s}")
            # 加入 Title Header
            print(f" {'─'*116}")
            print(f" {'操作指示':<10} | {'開賽時間':<6} | {'賽事 ID':<8} | {'聯賽':<6} | {'主隊':>10} vs {'客隊':<10} | {'系統建議讓分盤口':<16} | {'賠率':<5} | {'雙擎指標 (P:勝率 / D:淨勝)':<24} | {'盤口型態'}")
            print(f" {'─'*116}")
            cur_date = date_s

        mid = str(r.get("match_id", "?"))
        lg = str(r.get("league", "?"))[:6]
        home_s = str(r.get("home_team", r.get("home", "Home"))).strip()[:10]
        away_s = str(r.get("away_team", r.get("away", "Away"))).strip()[:10]
        
        pat = str(r.get('Pat_L', '-'))
        if pat == 'nan': pat = '-'
        pat = pat[:8]
        
        hc_flag = "🔥" if pat in ['凹陷誘盤', '雙峰擠壓', '反階梯'] else "  "
        cut_time = pd.to_datetime(r.get('line_dt_n1_T13H', None))
        cut_str = cut_time.strftime('%m-%d %H:%M') if pd.notnull(cut_time) else '-'
        if r["Action"] == "FAV":
            fav_home = (int(r.get("bet_fav_is_home", 1)) == 1)
            fav_team = home_s if fav_home else away_s
            fav_o = float(r.get("real_odds_fav", 0))
            hc_val = float(r.get("bet_fav_hc", 0))
            
            print(f" {hc_flag}[買 強隊] | {time_s:<8} | {mid:<10} | {lg:<6} | {home_s:>10} vs {away_s:<10} "
                  f"| 讓: {fav_team[:8]:8} 讓 {hc_val:<4.2f} "
                  f"| 賠: {fav_o:.3f} @ {cut_str} | P: {r['prob_Fav']:.3f}  D: +{r['pred_diff']:.2f} | {pat}")
        else:
            fav_home = (int(r.get("bet_fav_is_home", 1)) == 1)
            dog_team = away_s if fav_home else home_s
            dog_o = float(r.get("real_odds_dog", 0))
            hc_val = float(r.get("bet_fav_hc", 0))
            
            print(f" {hc_flag}[買 弱隊] | {time_s:<8} | {mid:<10} | {lg:<6} | {home_s:>10} vs {away_s:<10} "
                  f"| 受: {dog_team[:8]:8} 受 {hc_val:<4.2f} "
                  f"| 賠: {dog_o:.3f} "
                  f"| P: {r['prob_Dog']:.3f}  D: {r['pred_diff']:.2f} "
                  f"| {pat}")

    print(f"\n{'─'*120}")
    print(f" 📊 統計結算: 總出手={len(rec)} | 買強隊={sum(rec['Action']=='FAV')} | 買弱隊={sum(rec['Action']=='DOG')} | 🔥 金礦型態觸發={sum(rec['Pat_L'].isin(['凹陷誘盤', '雙峰擠壓', '反階梯']))}")
    print(f"{'='*120}")

    rec.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f" 📄 詳細推論清單已儲存至：{out_csv}")
    return rec

if __name__ == "__main__":
    print("="*85)
    print("🏆 V79 實盤雙擎快取系統 (最終黃金結案版)")
    print("="*85)

    TRAIN_M = "raw_features_merged.csv"
    TRAIN_L = "asian_lines_fixed_final.csv"
    FUT_M = "predict_eu15.csv"
    FUT_L = "predict_n1n415.csv"

    # 若要確保模型基底是最新的乾淨特徵，可保留 True 重訓一次，後續改回 False 即可秒速推論
    FORCE_RETRAIN = False 

    if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        print(f"✅ PKL 快取已掛載，版本：{bundle.get('version')} (建立於 {bundle.get('train_date')})")
    else:
        tr_df = prepare_dataset(TRAIN_M, TRAIN_L, is_train=True)
        if not tr_df.empty:
            bundle = train_and_save(tr_df, force_retrain=FORCE_RETRAIN)
        else:
            print("🚨 致命錯誤：無法訓練模型。")
            exit()

    if os.path.exists(FUT_M) and os.path.exists(FUT_L):
        print(f"\n📡 偵測到未來賽事資料，開始掃描...")
        fu_df = prepare_dataset(FUT_M, FUT_L, is_train=False)
        predict_and_print(bundle, fu_df, out_csv="ah_future_recommend.csv")
