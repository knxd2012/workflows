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

try:
    from curl_cffi import requests
    HAS_CFFI = True
    print("🚀 使用 curl_cffi (潛行模式已啟動)")
except ImportError:
    import requests
    HAS_CFFI = False
    print("ℹ️ 使用標準 requests 庫")

import V79_Core 

warnings.filterwarnings('ignore')

# ==================== 配置參數 ====================
WHITE_LIST = [
    "英甲", "巴西甲", "德乙", "挪超", "葡超", "瑞典超", "美職業", "阿甲", "英冠", "沙特聯",
    "英超", "荷乙", "蘇超", "德甲", "西乙", "芬超", "荷甲", "法乙", "西甲", "法甲",
    "意甲", "韓K聯", "日職聯", "日皇杯", "日職乙", "南美杯", "澳超", "解放者杯", "歐冠杯", "澳洲甲"
]

MODEL_PATH = "models_v79/AH_V79_DUAL_T13H.pkl"
FINAL_JSON = "web_results.json"
DIR_EU = "temp_eu"
DIR_AS = "temp_as2"

# ==================== 1. 爬蟲解析輔助函數 ====================
def parse_teams_eu(soup):
    title = soup.find('title')
    if not title: return 'Unknown Home', 'Unknown Away'
    teams = re.search(r'[:：]\s*(.+?)\s+VS\s+(.+?)(?:數據分析|$)', title.text)
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

# ==================== 2. 強力獲取 Match ID (移動網頁掃描法) ====================
def get_match_ids_stealth():
    """
    不再抓取 bf.js，改為直接掃描移動版首頁的 HTML
    這能有效避開對 JS 檔案的連線重置 (Error 56)
    """
    print("📡 啟動潛行掃描引擎...")
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1')
    
    driver = webdriver.Chrome(options=options)
    found_ids = []
    
    try:
        # 訪問移動版即時比分頁面
        url = "http://m.nowscore.com/"
        driver.get(url)
        time.sleep(8) # 給予充足的渲染時間
        
        # 嘗試從連結中提取 match_id (格式通常是 /detail/2345678.htm)
        html = driver.page_source
        matches = re.findall(r'detail/(\d+)\.htm', html)
        
        if matches:
            # 由於 HTML 不一定帶有聯賽名，我們先把所有發現的 ID 拿回來
            # 稍後在 EU 引擎爬取時再進行白名單過濾
            found_ids = list(set(matches))
            print(f"✨ 發現 {len(found_ids)} 場潛在賽事 ID")
        else:
            print("⚠️ 未能從 HTML 提取到賽事 ID，嘗試備援方案...")
            # 備援：抓取 JS (如果網頁載入了 JS)
            driver.get("http://m.nowscore.com/data/bf.js")
            time.sleep(3)
            content = driver.find_element(By.TAG_NAME, "body").text
            matches = re.findall(r'A\[\d+\]="(\d+)\^', content)
            found_ids = list(set(matches))
            print(f"✨ 備援引擎獲取到 {len(found_ids)} 個 ID")

    except Exception as e:
        print(f"❌ 潛行掃描報錯: {e}")
    finally:
        driver.quit()
    
    return found_ids

# ==================== 3. 爬蟲引擎 ====================
def scrape_eu_worker(match_chunk, progress, lock):
    batch = []
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            r = requests.get(url, timeout=10, http_version=1)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                league = parse_league_eu(soup)
                
                # 在這裡執行白名單過濾
                if not any(target in league for target in WHITE_LIST):
                    continue
                
                home, away = parse_teams_eu(soup)
                ko_time = parse_kickoff_time_eu(soup)
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows[1:]:
                        cols = row.find_all('td')
                        if len(cols) >= 6:
                            batch.append({
                                'match_id': mid, 'home_team': home, 'away_team': away,
                                'league': league, 'kickoff_time': ko_time,
                                'home_odds': cols[0].text.strip(), 'draw_odds': cols[1].text.strip(),
                                'away_odds': cols[2].text.strip(), 'change_time': cols[5].text.strip()
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
                        if len(cols) == 5:
                            batch.append({'match_id': mid, 'line': f'n{n}', 'home': cols[0].text, 'handicap': cols[1].text, 'away': cols[2].text, 'time': cols[3].text})
            except: continue
        with lock: progress['as'] += 1
    if batch: pd.DataFrame(batch).to_csv(os.path.join(DIR_AS, f"as_{random.randint(0,999)}.csv"), index=False)
    driver.quit()

# ==================== 4. 主程序 ====================
def main():
    os.makedirs(DIR_EU, exist_ok=True); os.makedirs(DIR_AS, exist_ok=True)
    
    # 第一步：獲取所有潛在 ID
    raw_ids = get_match_ids_stealth()
    if not raw_ids:
        print("📭 未能獲取任何賽事 ID。")
        return

    # 第二步：爬取並過濾
    lock = threading.Lock(); progress = {'eu': 0, 'as': 0}
    print(f"🚀 開始分析 {len(raw_ids)} 場潛在賽事...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(scrape_eu_worker, raw_ids, progress, lock)
        executor.submit(scrape_as_worker, raw_ids, progress, lock)
    
    # 合併 CSV
    for d, f in [(DIR_EU, "predict_eu16.csv"), (DIR_AS, "predict_n1n416.csv")]:
        files = glob.glob(os.path.join(d, "*.csv"))
        if files:
            pd.concat([pd.read_csv(x) for x in files]).to_csv(f, index=False)
            for x in files: os.remove(x)

    # 第三步：預測
    if os.path.exists(MODEL_PATH):
        print("🧠 啟動 V79 預測核心...")
        with open(MODEL_PATH, "rb") as f: bundle = pickle.load(f)
        try:
            fu_df = V79_Core.prepare_dataset("predict_eu16.csv", "predict_n1n416.csv", is_train=False)
            if not fu_df.empty:
                rec = V79_Core.predict_and_print(bundle, fu_df)
                web_data = {"update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')}
                with open(FINAL_JSON, 'w', encoding='utf-8') as f: json.dump(web_data, f, ensure_ascii=False, indent=4)
                print("🎉 預測成功！")
            else:
                print("⚠️ 沒有符合 T-13 條件的預測。")
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
    else:
        print(f"🚨 找不到模型檔案: {MODEL_PATH}")

if __name__ == "__main__":
    main()
