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

# ==================== 1. 修正匯入區塊 ====================
try:
    from curl_cffi import requests
    HAS_CFFI = True
    print("🚀 使用 curl_cffi (TLS 指紋偽裝已啟動)")
except ImportError:
    import requests
    HAS_CFFI = False
    print("ℹ️ 使用標準 requests 庫")

# 嘗試導入 V79 核心
try:
    import V79_Core
except ImportError:
    print("🚨 找不到 V79_Core.py，請確認檔名是否完全一致。")

warnings.filterwarnings('ignore')

# ==================== 2. 配置參數 ====================
WHITE_LIST = [
    "英甲", "巴西甲", "德乙", "挪超", "葡超", "瑞典超", "美职业", "阿甲", "英冠", "沙特联",
    "英超", "荷乙", "苏超", "德甲", "西乙", "芬超", "荷甲", "法乙", "西甲", "法甲",
    "意甲", "韩K联", "日职联", "日皇杯", "挪女超", "日職乙", "南美杯", "澳超", "解放者杯", "欧冠杯", "澳洲甲"
]

MAX_WORKERS_EU = 10
MAX_WORKERS_AS = 4

EU_OUTPUT_FILE = "predict_eu16.csv"
AS_OUTPUT_FILE = "predict_n1n416.csv"
FINAL_JSON = "web_results.json"
DIR_EU = "temp_eu"
DIR_AS = "temp_as2"
MODEL_PATH = "models_v79/AH_V79_DUAL_T13H.pkl"
USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"]

# ==================== 3. 解析函數 (輔助 V79_Core) ====================
def parse_teams_eu(soup):
    title = soup.find('title')
    if not title: return 'Unknown Home', 'Unknown Away'
    teams = re.search(r'[:：]\s*(.+?)\s+VS\s+(.+?)(?:数据分析|$)', title.text)
    if teams: return teams.group(1).strip(), teams.group(2).strip()
    return 'Unknown Home', 'Unknown Away'

def parse_league_eu(soup):
    top_div = soup.find('div', id='top')
    if top_div:
        line1_span = top_div.find('span', class_='line1')
        if line1_span: return line1_span.text.strip()
    return 'Unknown League'

def parse_kickoff_time_eu(soup):
    kickoff_tag = soup.find(string=re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}'))
    if kickoff_tag:
        try: return pd.to_datetime(kickoff_tag.strip())
        except: pass
    return pd.NaT

# ==================== 4. 自動獲取 Match ID (含重試與 HTTP 降級) ====================
def get_realtime_match_ids():
    timestamp = int(time.time() * 1000)
    url = f"https://live.nowscore.com/data/bf.js?{timestamp}"
    headers = {"Referer": "https://live.nowscore.com/", "User-Agent": random.choice(USER_AGENTS)}

    for attempt in range(3):
        try:
            print(f"📡 正在嘗試獲取賽事名單 (第 {attempt + 1} 次)...")
            if HAS_CFFI:
                # 強制使用 HTTP/1.1 避免 err 92 / err 8
                r = requests.get(url, headers=headers, timeout=15, impersonate="chrome120", allow_http2=False)
            else:
                r = requests.get(url, headers=headers, timeout=15)
            
            r.encoding = 'utf-8'
            content = r.text

            if "A[0]" not in content:
                time.sleep(2); continue

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
            print(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {e}")
            time.sleep(3)
    return []

# ==================== 5. 爬蟲 Worker ====================
def scrape_eu_worker(match_chunk, progress, lock):
    batch = []
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                home, away = parse_teams_eu(soup)
                league = parse_league_eu(soup)
                ko_time = parse_kickoff_time_eu(soup)
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
    if batch: pd.DataFrame(batch).to_csv(os.path.join(DIR_EU, f"eu_{random.randint(0,999)}.csv"), index=False)

def scrape_as_worker(match_chunk, progress, lock):
    options = Options()
    options.add_argument('--headless=new'); options.add_argument('--no-sandbox'); options.add_argument('--disable-dev-shm-usage')
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
                            batch.append({'match_id': mid, 'line': f'n{n}', 'home': cols[0].get_text(strip=True), 'handicap': cols[1].get_text(strip=True), 'away': cols[2].get_text(strip=True), 'time': cols[3].get_text(strip=True)})
            except: continue
        with lock: progress['as'] += 1
    if batch: pd.DataFrame(batch).to_csv(os.path.join(DIR_AS, f"as_{random.randint(0,999)}.csv"), index=False)
    driver.quit()

def merge_csv(temp_dir, output_file):
    files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if not files: return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    for f in files: os.remove(f)

# ==================== 6. 主流程指揮 ====================
def main():
    start_time = time.time()
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)
    
    ids = get_realtime_match_ids()
    if not ids:
        print("📭 今日無符合條件賽事，腳本結束。"); return
    
    lock = threading.Lock(); progress = {'eu': 0, 'as': 0}
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(scrape_eu_worker, ids, progress, lock)
        executor.submit(scrape_as_worker, ids, progress, lock)
    
    merge_csv(DIR_EU, EU_OUTPUT_FILE); merge_csv(DIR_AS, AS_OUTPUT_FILE)

    if os.path.exists(MODEL_PATH):
        print("🧠 執行 V79 預測核心...")
        with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
        fu_df = V79_Core.prepare_dataset(EU_OUTPUT_FILE, AS_OUTPUT_FILE, is_train=False)
        if not fu_df.empty:
            rec = V79_Core.predict_and_print(bundle, fu_df)
            web_data = {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')}
            with open(FINAL_JSON, 'w', encoding='utf-8') as f: json.dump(web_data, f, ensure_ascii=False, indent=4)
            print("🎉 預測結果生成成功！")
        else:
            print("⚠️ 未發現符合預測條件（T-13）的賽事數據。")
    else:
        print(f"🚨 模型檔案不存在：{MODEL_PATH}")

if __name__ == "__main__":
    main()
