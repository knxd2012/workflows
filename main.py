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

# ==================== 1. 環境與庫匯入 ====================
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
    print("✅ V79_Core 載入成功")
except ImportError:
    print("🚨 警告：找不到 V79_Core.py，請確保文件已上傳至根目錄。")

warnings.filterwarnings('ignore')

# ==================== 2. 全域配置 ====================
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

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]

# ==================== 3. 爬蟲解析輔助函數 ====================
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

# ==================== 4. 獲取 Match ID (深度備援) ====================

def selenium_fetch_backup(url):
    """
    備援引擎：當協定被封鎖時，模擬真實 Chrome 載入
    """
    print(f"🔄 啟動 Selenium 深度掃描: {url}")
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = webdriver.Chrome(options=options)
    # 執行 CDP 隱藏 webdriver 特徵
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

    content = ""
    try:
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(6) # 給予充足載入時間
        
        try:
            content = driver.find_element(By.TAG_NAME, "pre").text
        except:
            try:
                content = driver.find_element(By.TAG_NAME, "body").text
            except:
                content = driver.page_source
                
        if "A[" not in content:
            print("⚠️ Selenium 未能發現賽事陣列關鍵字")
    except Exception as e:
        print(f"❌ Selenium 備援執行報錯: {e}")
    finally:
        driver.quit()
    return content

def get_realtime_match_ids():
    timestamp = int(time.time() * 1000)
    # 多源備選地址
    sources = [
        f"https://live.nowscore.com/data/bf.js?{timestamp}",
        f"http://live.nowscore.com/data/bf.js?{timestamp}",
        f"https://livestatic.titan007.com/vbsxml/bfdata_ut.js?r={timestamp}"
    ]
    
    headers = {
        "Referer": "https://www.nowscore.com/",
        "User-Agent": random.choice(USER_AGENTS),
        "Connection": "keep-alive"
    }

    content = ""
    # 優先快速抓取
    for url in sources:
        try:
            print(f"📡 嘗試快速抓取: {url}")
            if HAS_CFFI:
                r = requests.get(url, headers=headers, timeout=15, impersonate="chrome120", http_version=1)
            else:
                r = requests.get(url, headers=headers, timeout=15)
            r.encoding = 'utf-8'
            if "A[" in r.text or "B[" in r.text:
                content = r.text
                print("✨ 快速抓取成功！")
                break
        except Exception as e:
            print(f"⚠️ 快速抓取失敗: {e}")

    # 若快速抓取全滅，啟動 Selenium
    if not content or "A[" not in content:
        for url in sources:
            content = selenium_fetch_backup(url)
            if content and ("A[" in content or "B[" in content):
                print("✨ Selenium 備援抓取成功！")
                break

    # 解析邏輯
    if content and ("A[" in content or "B[" in content):
        try:
            leagues_map = {}
            # 支援 B[i] = "..." 或 B[i] = [...]
            b_raw = re.findall(r'B\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
            for idx, val in b_raw:
                parts = val.replace("'", "").split('^')
                if len(parts) > 0: leagues_map[idx] = parts[0].strip()
            
            a_raw = re.findall(r'A\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
            final_ids = []
            for idx, val in a_raw:
                parts = [p.strip().strip("'") for p in val.split('^')] if "^" in val else [p.strip().strip("'") for p in val.split(',')]
                if len(parts) < 10: continue
                
                match_id = parts[0]
                league_idx = parts[1]
                league_name = leagues_map.get(league_idx, "")
                
                if any(target in league_name for target in WHITE_LIST):
                    final_ids.append(match_id)
            
            print(f"✅ 成功提取 {len(final_ids)} 個符合條件的 Match ID")
            return list(set(final_ids))
        except Exception as e:
            print(f"❌ 數據解析錯誤: {e}")
    
    return []

# ==================== 5. 爬蟲 Worker 邏輯 ====================

def scrape_eu_worker(match_chunk, progress, lock):
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    batch = []
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            r = session.get(url, timeout=10)
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
    if batch:
        pd.DataFrame(batch).to_csv(os.path.join(DIR_EU, f"eu_{random.randint(0,999)}.csv"), index=False)

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
                WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, "//table[@bgcolor='#AFC7E2']//tr[position()>1]")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table', {'cellspacing': '1', 'bgcolor': '#AFC7E2'})
                if table:
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) == 5 and cols[4].get_text(strip=True) != '滚':
                            batch.append({'match_id': mid, 'line': f'n{n}', 'home': cols[0].get_text(strip=True), 'handicap': cols[1].get_text(strip=True), 'away': cols[2].get_text(strip=True), 'time': cols[3].get_text(strip=True)})
            except: continue
        with lock: progress['as'] += 1
    if batch:
        pd.DataFrame(batch).to_csv(os.path.join(DIR_AS, f"as_{random.randint(0,999)}.csv"), index=False)
    driver.quit()

def merge_csv(temp_dir, output_file):
    files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if not files: return
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    for f in files: os.remove(f)

# ==================== 6. 主執行入口 ====================

def main():
    start_time = time.time()
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)
    
    # 1. 獲取 ID
    match_ids = get_realtime_match_ids()
    if not match_ids:
        print("📭 今日無白名單內賽事或獲取失敗。")
        return

    print(f"🔥 已鎖定 {len(match_ids)} 場賽事，啟動非同步雙軌爬蟲...")
    lock = threading.Lock()
    progress = {'eu': 0, 'as': 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EU + MAX_WORKERS_AS) as executor:
        eu_chunks = np.array_split(match_ids, MAX_WORKERS_EU) if len(match_ids) >= MAX_WORKERS_EU else [match_ids]
        for chunk in eu_chunks:
            executor.submit(scrape_eu_worker, chunk.tolist(), progress, lock)
        
        as_chunks = np.array_split(match_ids, MAX_WORKERS_AS) if len(match_ids) >= MAX_WORKERS_AS else [match_ids]
        for chunk in as_chunks:
            executor.submit(scrape_as_worker, chunk.tolist(), progress, lock)

    # 2. 合併數據
    print("📦 合併爬蟲結果...")
    merge_csv(DIR_EU, EU_OUTPUT_FILE)
    merge_csv(DIR_AS, AS_OUTPUT_FILE)

    # 3. 推論
    if os.path.exists(MODEL_PATH):
        print("🧠 啟動 V79 黃金推論系統...")
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            
            # 調用 V79_Core 的預測邏輯
            fu_df = V79_Core.prepare_dataset(EU_OUTPUT_FILE, AS_OUTPUT_FILE, is_train=False)
            
            if not fu_df.empty:
                rec = V79_Core.predict_and_print(bundle, fu_df)
                
                web_data = {
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')
                }
                with open(FINAL_JSON, 'w', encoding='utf-8') as f:
                    json.dump(web_data, f, ensure_ascii=False, indent=4)
                print(f"🎉 成功！預測結果已同步至 {FINAL_JSON}")
            else:
                print("⚠️ 今日數據不符合 V79 策略門檻 (T-13h)。")
        except Exception as e:
            print(f"❌ V79 推論失敗: {e}")
    else:
        print(f"🚨 錯誤：找不到模型 {MODEL_PATH}")

    print(f"⌛ 耗時: {(time.time() - start_time)/60:.2f} 分鐘")

if __name__ == "__main__":
    main()
