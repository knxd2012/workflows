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
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==================== 1. 環境判斷與核心導入 ====================
try:
    from curl_cffi import requests
    HAS_CFFI = True
    print("🚀 使用 curl_cffi (潛行模式已啟動)")
except ImportError:
    import requests
    HAS_CFFI = False
    print("ℹ️ 使用標準 requests 庫")

try:
    import V79_Core
    print("✅ V79_Core 載入成功")
except ImportError:
    print("🚨 錯誤：找不到 V79_Core.py，請確認檔案已在目前目錄下。")

warnings.filterwarnings('ignore')

# ==================== 2. 全域配置 ====================
WHITE_LIST = [
    "英甲", "巴西甲", "德乙", "挪超", "葡超", "瑞典超", "美职业", "阿甲", "英冠", "沙特联",
    "英超", "荷乙", "苏超", "德甲", "西乙", "芬超", "荷甲", "法乙", "西甲", "法甲",
    "意甲", "韩K联", "日职联", "日皇杯", "日職乙", "南美杯", "澳超", "解放者杯", "欧冠杯", "澳洲甲"
]

MODEL_PATH = "models_v79/AH_V79_DUAL_T13H.pkl"
FINAL_JSON = "web_results.json"
DIR_EU = "temp_eu"
DIR_AS = "temp_as2"
USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"]

# ==================== 3. Colab 專用 Driver 設定 ====================
def create_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-blink-features=AutomationControlled')
    # 指向 Colab 安裝的 Chromium
    options.binary_location = "/usr/bin/chromium-browser"
    options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
    
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

# ==================== 4. 獲取 Match ID (潛行網頁掃描法) ====================
def get_realtime_match_ids():
    print("📡 啟動潛行掃描引擎獲取 Match ID...")
    driver = create_driver()
    found_ids = []
    
    try:
        # 直接掃描手機版首頁，避開 API 封鎖
        driver.get("http://m.nowscore.com/")
        time.sleep(8) # 等待渲染
        
        # 從 HTML 提取 detail 連結中的 ID
        html = driver.page_source
        matches = re.findall(r'detail/(\d+)\.htm', html)
        
        if not matches:
            # 備援：如果首頁抓不到，嘗試直接抓取數據 js
            timestamp = int(time.time() * 1000)
            driver.get(f"http://m.nowscore.com/data/bf.js?{timestamp}")
            time.sleep(3)
            content = driver.find_element(By.TAG_NAME, "body").text
            matches = re.findall(r'A\[\d+\]="(\d+)\^', content)
            
        found_ids = list(set(matches))
        print(f"✨ 共發現 {len(found_ids)} 場潛在賽事 ID")
    except Exception as e:
        print(f"❌ 獲取 ID 失敗: {e}")
    finally:
        driver.quit()
    return found_ids

# ==================== 5. 爬蟲 Worker 邏輯 ====================
def parse_teams_eu(soup):
    title = soup.find('title')
    if not title: return 'Unknown', 'Unknown'
    teams = re.search(r'[:：]\s*(.+?)\s+VS\s+(.+?)(?:数据分析|$)', title.text)
    return (teams.group(1).strip(), teams.group(2).strip()) if teams else ('Unknown', 'Unknown')

def scrape_eu_worker(match_chunk, progress, lock):
    batch = []
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            # 使用 http_version=1 解決 HTTP/2 stream 錯誤
            r = requests.get(url, timeout=12, impersonate="chrome120", http_version=1) if HAS_CFFI else requests.get(url, timeout=12)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # 獲取聯賽並過濾白名單
                top_div = soup.find('div', id='top')
                league = top_div.find('span', class_='line1').text.strip() if top_div else "Unknown"
                
                if any(target in league for target in WHITE_LIST):
                    home, away = parse_teams_eu(soup)
                    ko_tag = soup.find(string=re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}'))
                    for table in soup.find_all('table'):
                        rows = table.find_all('tr')[1:]
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) >= 6:
                                batch.append({
                                    'match_id': mid, 'home_team': home, 'away_team': away,
                                    'league': league, 'kickoff_time': ko_tag.strip() if ko_tag else "",
                                    'home_odds': cols[0].text.strip(), 'draw_odds': cols[1].text.strip(),
                                    'away_odds': cols[2].text.strip(), 'change_time': cols[5].text.strip()
                                })
            with lock: progress['eu'] += 1
        except: continue
    if batch: pd.DataFrame(batch).to_csv(os.path.join(DIR_EU, f"eu_{random.randint(0,999)}.csv"), index=False)

def scrape_as_worker(match_chunk, progress, lock):
    driver = create_driver()
    batch = []
    for mid in match_chunk:
        for n in range(1, 5):
            url = f"https://vip.titan007.com/changeDetail/multiHandicap.aspx?id={mid}&companyID=47&n={n}"
            try:
                driver.get(url)
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//table[@bgcolor='#AFC7E2']")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table', {'bgcolor': '#AFC7E2'})
                if table:
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) == 5:
                            batch.append({'match_id': mid, 'line': f'n{n}', 'home': cols[0].text, 'handicap': cols[1].text, 'away': cols[2].text, 'time': cols[3].text})
            except: continue
        with lock: progress['as'] += 1
    if batch: pd.DataFrame(batch).to_csv(os.path.join(DIR_AS, f"as_{random.randint(0,999)}.csv"), index=False)
    driver.quit()

# ==================== 6. 主指揮中心 ====================
def main():
    start_time = time.time()
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)
    
    # 1. 掃描 ID
    raw_ids = get_realtime_match_ids()
    if not raw_ids:
        print("📭 目前無賽事數據。"); return

    # 2. 啟動爬蟲
    print(f"🚀 開始分析 {len(raw_ids)} 場賽事...")
    lock = threading.Lock(); progress = {'eu': 0, 'as': 0}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(scrape_eu_worker, raw_ids, progress, lock)
        executor.submit(scrape_as_worker, raw_ids, progress, lock)
    
    # 3. 合併結果
    for d, f in [(DIR_EU, "predict_eu16.csv"), (DIR_AS, "predict_n1n416.csv")]:
        files = glob.glob(os.path.join(d, "*.csv"))
        if files:
            pd.concat([pd.read_csv(x) for x in files]).to_csv(f, index=False)
            for x in files: os.remove(x)

    # 4. 推論
    if os.path.exists(MODEL_PATH):
        print("🧠 執行 V79 預測模型...")
        with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
        try:
            fu_df = V79_Core.prepare_dataset("predict_eu16.csv", "predict_n1n416.csv", is_train=False)
            if not fu_df.empty:
                rec = V79_Core.predict_and_print(bundle, fu_df)
                web_data = {
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')
                }
                with open(FINAL_JSON, 'w', encoding='utf-8') as f:
                    json.dump(web_data, f, ensure_ascii=False, indent=4)
                print(f"🎉 預測完成，結果已存入 {FINAL_JSON}")
            else:
                print("⚠️ 無符合條件賽事。")
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
    else:
        print(f"🚨 找不到模型：{MODEL_PATH}")

    print(f"⌛ 耗時: {(time.time() - start_time)/60:.2f} 分鐘")

if __name__ == "__main__":
    main()
