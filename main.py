import pandas as pd
import numpy as np
import os
import re
import json
import time
import random
import threading
import glob
import warnings
import pickle
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 嘗試導入 TLS 偽裝套件
try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False
    import requests

# 導入 V79 核心邏輯
try:
    import V79_Core
except ImportError:
    print("🚨 錯誤：找不到 V79_Core.py，請確保文件已上傳並更名正確。")

warnings.filterwarnings('ignore')

# ==================== 配置與路徑設定 ====================
WHITE_LIST = [
    "英甲", "巴西甲", "德乙", "挪超", "葡超", "瑞典超", "美职业", "阿甲", "英冠", "沙特联",
    "英超", "荷乙", "苏超", "德甲", "西乙", "芬超", "荷甲", "法乙", "西甲", "法甲",
    "意甲", "韩K联", "日职联", "日皇杯", "挪女超", "日職乙", "南美杯", "国际友谊", "北美预选",
    "澳超", "南美预选", "智利甲", "南非洲杯", "解放者杯", "欧洲预选", "欧女国联", "俄超降",
    "苏冠附", "沙王冠", "俄超", "瑞典女超", "俄杯", "比甲冠", "法乙升", "比甲附", "欧青U21",
    "荷甲附", "德乙升", "欧女杯", "欧冠杯", "阿根廷杯", "美金杯", "葡杯",
    "荷乙附", "墨西甲附", "世俱杯", "法國杯", "亞女冠杯", "智利杯", "德國杯", "美冠杯", "欧會杯", "澳洲甲"
]

MAX_WORKERS_EU = 10
MAX_WORKERS_AS = 4

EU_OUTPUT_FILE = "predict_eu16.csv"
AS_OUTPUT_FILE = "predict_n1n416.csv"
FINAL_JSON = "web_results.json"
DIR_EU = "temp_eu"
DIR_AS = "temp_as2"
MODEL_PATH = "models_v79/AH_V79_DUAL_T13H.pkl"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
]

# ==================== 1. 自動獲取 Match ID ====================
def get_realtime_match_ids():
    timestamp = int(time.time() * 1000)
    url = f"https://live.nowscore.com/data/bf.js?{timestamp}"
    headers = {"Referer": "https://live.nowscore.com/", "User-Agent": random.choice(USER_AGENTS)}
    try:
        print(f"📡 正在從 Nowscore 獲取即時賽事名單...")
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = 'utf-8'
        content = r.text

        leagues_map = {}
        b_raw = re.findall(r'B\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        for idx, val in b_raw:
            parts = val.replace("'", "").split('^')
            if len(parts) > 0: leagues_map[idx] = parts[0].strip()
        
        a_raw = re.findall(r'A\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        final_ids = []
        for idx, val in a_raw:
            parts = [p.strip().strip("'") for p in val.split('^')] if "^" in val else [p.strip().strip("'") for p in val.split(',')]
            if len(parts) < 10: continue
            if leagues_map.get(parts[1], "") in WHITE_LIST:
                final_ids.append(parts[0])
        
        print(f"✅ 成功獲取 {len(final_ids)} 場白名單賽事 ID")
        return list(set(final_ids))
    except Exception as e:
        print(f"❌ 獲取 ID 失敗: {e}")
        return []

# ==================== 2. 爬蟲引擎邏輯 ====================
def create_http_session():
    ua = random.choice(USER_AGENTS)
    s = cffi_requests.Session(impersonate="chrome120") if HAS_CFFI else requests.Session()
    s.headers.update({"User-Agent": ua})
    return s

def scrape_eu_worker(match_chunk, progress, lock):
    session = create_http_session()
    batch = []
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # 這裡調用 V79_Core 內的解析函數
                home, away = V79_Core.parse_teams_eu(soup)
                league = V79_Core.parse_league_eu(soup)
                ko_time = V79_Core.parse_kickoff_time_eu(soup)
                
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    if len(rows) < 2: continue
                    for row in rows[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 6:
                            batch.append({
                                'match_id': mid, 'home_team': home, 'away_team': away,
                                'league': league, 'kickoff_time': ko_time,
                                'home_odds': cols[0].text.strip(), 'draw_odds': cols[1].text.strip(),
                                'away_odds': cols[2].text.strip(), 'return_rate': cols[3].text.strip(),
                                'change_time': cols[5].text.strip()
                            })
            with lock: progress['eu'] += 1
        except: continue
    
    if batch:
        pd.DataFrame(batch).to_csv(os.path.join(DIR_EU, f"batch_{random.randint(100,999)}.csv"), index=False)

def scrape_as_worker(match_chunk, progress, lock):
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    
    batch = []
    for mid in match_chunk:
        for n in range(1, 5):
            url = f"https://vip.titan007.com/changeDetail/multiHandicap.aspx?id={mid}&companyID=47&n={n}"
            try:
                driver.get(url)
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//table[@bgcolor='#AFC7E2']//tr[position()>1]")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table', {'cellspacing': '1', 'bgcolor': '#AFC7E2'})
                if table:
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) == 5 and cols[4].get_text(strip=True) != '滚':
                            batch.append({
                                'match_id': mid, 'line': f'n{n}',
                                'home': cols[0].get_text(strip=True),
                                'handicap': cols[1].get_text(strip=True),
                                'away': cols[2].get_text(strip=True),
                                'time': cols[3].get_text(strip=True)
                            })
            except: continue
        with lock: progress['as'] += 1
    
    if batch:
        pd.DataFrame(batch).to_csv(os.path.join(DIR_AS, f"batch_{random.randint(100,999)}.csv"), index=False)
    driver.quit()

def merge_csv(temp_dir, output_file):
    files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if not files: return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    for f in files: os.remove(f)

# ==================== 3. 主執行流程 ====================
def main():
    start_time = time.time()
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)

    # A. 獲取賽事
    match_ids = get_realtime_match_ids()
    if not match_ids:
        print("📭 今日無符合條件賽事，腳本結束。"); return

    # B. 多線程爬取
    lock = threading.Lock()
    progress = {'eu': 0, 'as': 0}
    
    print(f"🚀 啟動雙軌爬蟲引擎 (EU x{MAX_WORKERS_EU}, AS x{MAX_WORKERS_AS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EU + MAX_WORKERS_AS) as executor:
        # EU 分組
        eu_chunks = np.array_split(match_ids, MAX_WORKERS_EU) if len(match_ids) >= MAX_WORKERS_EU else [match_ids]
        for chunk in eu_chunks:
            executor.submit(scrape_eu_worker, chunk.tolist(), progress, lock)
        
        # AS 分組
        as_chunks = np.array_split(match_ids, MAX_WORKERS_AS) if len(match_ids) >= MAX_WORKERS_AS else [match_ids]
        for chunk in as_chunks:
            executor.submit(scrape_as_worker, chunk.tolist(), progress, lock)

    # C. 合併數據
    print("📦 正在合併臨時數據...")
    merge_csv(DIR_EU, EU_OUTPUT_FILE)
    merge_csv(DIR_AS, AS_OUTPUT_FILE)

    # D. V79 預測推論
    if os.path.exists(MODEL_PATH):
        print("🧠 執行 V79 雙擎推論核心...")
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            
            # 準備數據集
            fu_df = V79_Core.prepare_dataset(EU_OUTPUT_FILE, AS_OUTPUT_FILE, is_train=False)
            
            if not fu_df.empty:
                # 執行預測
                rec = V79_Core.predict_and_print(bundle, fu_df, out_csv="ah_recommend_v79.csv")
                
                if not rec.empty:
                    # 生成網頁用 JSON
                    web_data = {
                        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')
                    }
                    with open(FINAL_JSON, 'w', encoding='utf-8') as f:
                        json.dump(web_data, f, ensure_ascii=False, indent=4)
                    print(f"🎉 預測完成！結果已存入 {FINAL_JSON}")
            else:
                print("⚠️ 預測數據集為空，可能爬蟲未獲取到有效 T-13 數據。")
        except Exception as e:
            print(f"❌ 預測階段發生錯誤: {e}")
    else:
        print(f"🚨 錯誤：找不到模型文件 {MODEL_PATH}")

    print(f"⌛ 總耗時: {(time.time() - start_time)/60:.2f} 分鐘")

if __name__ == "__main__":
    main()
