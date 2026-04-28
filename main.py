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

# ==================== 1. 環境判斷與庫導入 ====================
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

# ==================== 2. 全域配置與白名單 ====================
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

# ==================== 3. Colab 專用 Driver 配置 (核心修復) ====================
def create_driver():
    options = Options()
    # --- Colab 必須參數 ---
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.binary_location = "/usr/bin/chromium-browser" # 指向 Colab 的二進制文件
    
    # --- 潛行偽裝 ---
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
    
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

# ==================== 4. 獲取 Match ID (直接掃描 bf.js) ====================
def get_realtime_match_ids():
    """
    透過 Selenium 直接開啟 bf.js 數據源
    這是避開 404/Error 56 獲取 Match ID 最穩定的做法
    """
    timestamp = int(time.time() * 1000)
    url = f"http://live.nowscore.com/data/bf.js?{timestamp}"
    
    print(f"📡 正在從數據源獲取賽事 (bf.js)...")
    driver = create_driver()
    final_ids = []
    
    try:
        driver.get(url)
        time.sleep(5) # 等待文本加載
        
        # 在瀏覽器中，JS 文件內容通常被包裹在 <pre> 標籤內
        try:
            content = driver.find_element(By.TAG_NAME, "pre").text
        except:
            content = driver.page_source
            
        if "A[" not in content:
            print("⚠️ 數據源為空或被攔截")
            return []

        # 解析聯賽 B 陣列
        leagues_map = {}
        b_raw = re.findall(r'B\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        for idx, val in b_raw:
            parts = val.replace("'", "").split('^')
            if len(parts) > 0: leagues_map[idx] = parts[0].strip()
        
        # 解析賽事 A 陣列
        a_raw = re.findall(r'A\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        for idx, val in a_raw:
            # 判斷是逗號分隔還是尖括號分隔
            parts = [p.strip().strip("'") for p in val.split('^')] if "^" in val else [p.strip().strip("'") for p in val.split(',')]
            
            if len(parts) < 10: continue
            
            match_id = parts[0]
            league_name = leagues_map.get(parts[1], "")
            
            # 過濾白名單
            if any(target in league_name for target in WHITE_LIST):
                final_ids.append(match_id)
        
        print(f"✅ 成功提取 {len(final_ids)} 場白名單賽事 ID")
    except Exception as e:
        print(f"❌ 獲取 bf.js 失敗: {e}")
    finally:
        driver.quit()
    return list(set(final_ids))

# ==================== 5. 爬蟲 Worker ====================
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
            # 獲取詳情頁
            r = requests.get(url, timeout=12, impersonate="chrome120", http_version=1) if HAS_CFFI else requests.get(url, timeout=12)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                home, away = parse_teams_eu(soup)
                top_div = soup.find('div', id='top')
                league = top_div.find('span', class_='line1').text.strip() if top_div else "Unknown"
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
                WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, "//table[@bgcolor='#AFC7E2']")))
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

# ==================== 6. 主程序入口 ====================
def main():
    start_time = time.time()
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)
    
    # 1. 獲取 Match IDs
    ids = get_realtime_match_ids()
    if not ids:
        print("📭 目前無符合條件之賽事。"); return

    # 2. 啟動爬蟲引擎
    print(f"🚀 開始分析 {len(ids)} 場精選賽事...")
    lock = threading.Lock(); progress = {'eu': 0, 'as': 0}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(scrape_eu_worker, ids, progress, lock)
        executor.submit(scrape_as_worker, ids, progress, lock)
    
    # 3. 合併數據 (CSV)
    eu_files = glob.glob(os.path.join(DIR_EU, "*.csv"))
    if eu_files: pd.concat([pd.read_csv(f) for f in eu_files]).to_csv("predict_eu16.csv", index=False)
    
    as_files = glob.glob(os.path.join(DIR_AS, "*.csv"))
    if as_files: pd.concat([pd.read_csv(f) for f in as_files]).to_csv("predict_n1n416.csv", index=False)
    
    # 4. 推論結果
    if os.path.exists(MODEL_PATH):
        print("🧠 啟動 V79 黃金預測模組...")
        with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
        try:
            # 呼叫你上傳的 V79_Core 原版邏輯
            fu_df = V79_Core.prepare_dataset("predict_eu16.csv", "predict_n1n416.csv", is_train=False)
            if not fu_df.empty:
                rec = V79_Core.predict_and_print(bundle, fu_df)
                # 存成網頁用 JSON
                web_data = {
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')
                }
                with open(FINAL_JSON, 'w', encoding='utf-8') as f:
                    json.dump(web_data, f, ensure_ascii=False, indent=4)
                print(f"🎉 成功！結果已存至 {FINAL_JSON}")
            else:
                print("⚠️ 無符合 V79 門檻條件之數據。")
        except Exception as e:
            print(f"❌ 預測階段出錯: {e}")
    else:
        print(f"🚨 找不到模型：{MODEL_PATH}")

    print(f"⌛ 耗時: {(time.time() - start_time)/60:.2f} 分鐘")

if __name__ == "__main__":
    main()
