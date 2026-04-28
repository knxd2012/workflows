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

# ==================== 終極 TLS 指紋偽裝套件 ====================
try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False
    import requests

# 屏蔽警告
warnings.filterwarnings('ignore')

# ==================== 1. 全域參數與白名單 ====================
WHITE_LIST = [
    "英甲", "巴西甲", "德乙", "挪超", "葡超", "瑞典超", "美职业", "阿甲", "英冠", "沙特联",
    "英超", "荷乙", "苏超", "德甲", "西乙", "芬超", "荷甲", "法乙", "西甲", "法甲",
    "意甲", "韩K联", "日职联", "日皇杯", "挪女超", "日職乙", "南美杯", "国际友谊", "北美预选",
    "澳超", "南美预选", "智利甲", "南非洲杯", "解放者杯", "欧洲预选", "欧女国联", "俄超降",
    "苏冠附", "沙王冠", "俄超", "瑞典女超", "俄杯", "比甲冠", "法乙升", "比甲附", "欧青U21",
    "荷甲附", "德乙升", "欧女杯", "芬兰杯", "巴高联", "欧冠杯", "阿根廷杯", "美金杯", "葡杯",
    "荷乙附", "墨西甲附", "世俱杯", "法国杯", "亚女冠杯", "智利杯", "德国杯", "美冠杯", "欧會杯", "澳洲甲"
]

MAX_WORKERS_EU = 10     # EU HTTP 引擎
MAX_WORKERS_AS = 4      # AS Selenium 引擎

EU_OUTPUT_FILE = "predict_eu16.csv"
AS_OUTPUT_FILE = "predict_n1n416.csv"
FINAL_JSON = "web_results.json"

DIR_EU = "temp_eu"
DIR_AS = "temp_as2"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
]

# ==================== 2. 自動獲取 Match ID (Nowscore bf.js) ====================
def get_realtime_match_ids():
    timestamp = int(time.time() * 1000)
    url = f"https://live.nowscore.com/data/bf.js?{timestamp}"
    headers = {
        "Referer": "https://live.nowscore.com/",
        "User-Agent": random.choice(USER_AGENTS)
    }
    try:
        print(f"📡 正在獲取最新賽事名單 (Timestamp: {timestamp})...")
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = 'utf-8'
        content = r.text

        # 提取聯賽 B 陣列
        leagues_map = {}
        b_raw = re.findall(r'B\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        for idx, val in b_raw:
            parts = val.replace("'", "").split('^')
            if len(parts) > 0: leagues_map[idx] = parts[0].strip()

        # 提取賽事 A 陣列
        a_raw = re.findall(r'A\[(\d+)\]\s*=\s*[\"\[](.*?)[\"\]];', content)
        final_ids = []
        for idx, val in a_raw:
            if "^" in val:
                parts = [p.strip().strip("'") for p in val.split('^')]
            else:
                parts = [p.strip().strip("'") for p in val.split(',')]
            
            if len(parts) < 10: continue
            match_id, league_idx = parts[0], parts[1]
            league_name = leagues_map.get(league_idx, "")

            if any(target == league_name for target in WHITE_LIST):
                final_ids.append(match_id)
        
        print(f"✅ 成功獲取 {len(final_ids)} 場白名單賽事 ID")
        return list(set(final_ids))
    except Exception as e:
        print(f"❌ 獲取 ID 失敗: {e}")
        return []

# ==================== 3. 爬蟲引擎公用解析函數 ====================
def parse_timestamp(change_time_raw, kickoff_time):
    if not change_time_raw: return pd.NaT
    change_time_raw = str(change_time_raw).strip()
    year = None
    if pd.notna(kickoff_time):
        try: year = kickoff_time.year
        except: pass
    if year:
        try: return pd.to_datetime(f"{year}-{change_time_raw}", format="%Y-%m-%d %H:%M", errors='coerce')
        except: pass
    try: return pd.to_datetime(change_time_raw, format='%Y-%m-%d %H:%M', errors='coerce')
    except: pass
    return change_time_raw

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

# ==================== 4. EU HTTP 引擎 ====================
def create_http_session():
    ua = random.choice(USER_AGENTS)
    if HAS_CFFI:
        s = cffi_requests.Session(impersonate="chrome120")
        s.headers.update({"User-Agent": ua})
        return s
    else:
        s = requests.Session()
        s.headers.update({"User-Agent": ua})
        return s

def scrape_euro_odds_http(match_id, session):
    url = f"https://m.nowscore.com/1x2Detail/{match_id}_177.htm"
    for attempt in range(4): 
        try:
            r = session.get(url, timeout=12)
            if r.status_code != 200:
                time.sleep(1.5); continue
            html = r.text
            if "</table>" not in html.lower():
                time.sleep(1.5); continue
            soup = BeautifulSoup(html, 'html.parser')
            home_team, away_team = parse_teams_eu(soup)
            league = parse_league_eu(soup)
            kickoff_time = parse_kickoff_time_eu(soup)
            
            records = []
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                if len(rows) < 2: continue 
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        try:
                            home_odds = float(cols[0].text.strip() or 0)
                            draw_odds = float(cols[1].text.strip() or 0)
                            away_odds = float(cols[2].text.strip() or 0)
                            return_rate = float(cols[3].text.strip().replace('%', '') or 0)
                            kelly_text = cols[4].text.strip()
                            kelly_parts = [x.strip() for x in kelly_text.split('\n') if x.strip()]
                            home_kelly = float(kelly_parts[0]) if len(kelly_parts) > 0 else 0
                            draw_kelly = float(kelly_parts[1]) if len(kelly_parts) > 1 else 0
                            away_kelly = float(kelly_parts[2]) if len(kelly_parts) > 2 else 0
                            change_time_raw = cols[5].text.strip()
                            records.append({
                                'home_team': home_team, 'away_team': away_team,
                                'kickoff_time': kickoff_time, 'league': league,
                                'home_odds': home_odds, 'draw_odds': draw_odds,
                                'away_odds': away_odds, 'return_rate': return_rate,
                                'home_kelly': home_kelly, 'draw_kelly': draw_kelly,
                                'away_kelly': away_kelly,
                                'change_time': parse_timestamp(change_time_raw, kickoff_time),
                                'match_id': match_id
                            })
                        except: continue
            if records: return pd.DataFrame(records)
            else: time.sleep(1)
        except: time.sleep(random.uniform(0.5, 1.5))
    return pd.DataFrame()

def eu_worker_thread(match_chunk, progress, lock):
    session = create_http_session()
    batch_data = []
    try:
        for mid in match_chunk:
            df = scrape_euro_odds_http(mid, session)
            if not df.empty: batch_data.append(df)
            with lock: progress['eu_completed'] += 1
            if len(batch_data) >= 10:
                with lock:
                    df_save = pd.concat(batch_data, ignore_index=True)
                    df_save.to_csv(os.path.join(DIR_EU, f"{mid}_eu.csv"), index=False, encoding='utf-8-sig')
                batch_data = []
    finally:
        if batch_data:
            with lock:
                df_save = pd.concat(batch_data, ignore_index=True)
                df_save.to_csv(os.path.join(DIR_EU, f"final_{random.randint(100,999)}_eu.csv"), index=False, encoding='utf-8-sig')
        if HAS_CFFI: session.close()

# ==================== 5. AS Selenium 引擎 ====================
def create_driver():
    options = Options()
    options.add_argument('--headless=new')  
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15) 
    return driver

def as_worker_thread(match_chunk, progress, lock):
    driver = None
    batch_data = []
    try:
        driver = create_driver()
        for match_id in match_chunk:
            match_lines = []
            for n in range(1, 5):
                url = f"https://vip.titan007.com/changeDetail/multiHandicap.aspx?id={match_id}&companyID=47&n={n}"
                success = False
                for attempt in range(3):
                    try:
                        driver.get(url)
                        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, "//table[@bgcolor='#AFC7E2']//tr[position()>1]")))
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        table = soup.find('table', {'cellspacing': '1', 'bgcolor': '#AFC7E2'})
                        if not table: raise ValueError("Table not found")
                        records = []
                        for row in table.find_all('tr')[1:]:
                            cols = row.find_all('td')
                            if len(cols) == 5 and cols[4].get_text(strip=True) != '滚':
                                records.append({
                                    'match_id': match_id, 'line': f'n{n}',
                                    'home': float(cols[0].get_text(strip=True) or 0),
                                    'handicap': cols[1].get_text(strip=True),
                                    'away': float(cols[2].get_text(strip=True) or 0),
                                    'time': cols[3].get_text(strip=True)
                                })
                        if records:
                            match_lines.extend(records); success = True; break
                    except: time.sleep(1.5)
            if match_lines: batch_data.append(pd.DataFrame(match_lines))
            with lock: progress['as_completed'] += 1
            if len(batch_data) >= 5:
                with lock:
                    df_save = pd.concat(batch_data, ignore_index=True)
                    df_save.to_csv(os.path.join(DIR_AS, f"{match_id}_as.csv"), index=False, encoding='utf-8-sig')
                batch_data = []
    finally:
        if batch_data:
            with lock:
                df_save = pd.concat(batch_data, ignore_index=True)
                df_save.to_csv(os.path.join(DIR_AS, f"final_{random.randint(100,999)}_as.csv"), index=False, encoding='utf-8-sig')
        if driver: driver.quit()

# ==================== 6. 合併模組 ====================
def overwrite_merge_temp_files(temp_dir, output_file, sort_cols=None):
    temp_files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if not temp_files: return
    df_list = []
    for f in temp_files:
        try:
            df = pd.read_csv(f, dtype={'match_id': str})
            if not df.empty: df_list.append(df)
        except: pass
    if not df_list: return
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.drop_duplicates(keep='last', inplace=True)
    if sort_cols: final_df.sort_values(by=sort_cols, inplace=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    for f in temp_files: os.remove(f)

# ==================== 7. 指揮中心 ====================
def main():
    os.makedirs(DIR_EU, exist_ok=True)
    os.makedirs(DIR_AS, exist_ok=True)

    # 1. 自動獲取 Match IDs
    match_ids = get_realtime_match_ids()
    if not match_ids:
        print("📭 今日無符合條件賽事，腳本結束")
        return

    lock = threading.Lock()
    progress = {'eu_completed': 0, 'eu_total': len(match_ids), 'as_completed': 0, 'as_total': len(match_ids)}

    # 2. 啟動 EU 引擎
    eu_chunk_size = (len(match_ids) + MAX_WORKERS_EU - 1) // MAX_WORKERS_EU
    eu_chunks = [match_ids[i:i + eu_chunk_size] for i in range(0, len(match_ids), eu_chunk_size)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EU) as executor:
        for chunk in eu_chunks: executor.submit(eu_worker_thread, chunk, progress, lock)
    overwrite_merge_temp_files(DIR_EU, EU_OUTPUT_FILE)

    # 3. 啟動 AS 引擎
    as_chunk_size = (len(match_ids) + MAX_WORKERS_AS - 1) // MAX_WORKERS_AS
    as_chunks = [match_ids[i:i + as_chunk_size] for i in range(0, len(match_ids), as_chunk_size)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_AS) as executor:
        for chunk in as_chunks: executor.submit(as_worker_thread, chunk, progress, lock)
    overwrite_merge_temp_files(DIR_AS, AS_OUTPUT_FILE)

    # 4. 呼叫 V79 預測模型
    print("🧠 啟動 V79 預測推論...")
    try:
        # 注意：此處需確保 V79 腳本內有 predict_and_print 函數
        import V79_Core 
        rec = V79_Core.predict_and_print(FUT_M=EU_OUTPUT_FILE, FUT_L=AS_OUTPUT_FILE)
        
        if not rec.empty:
            # 轉換為網頁 JSON
            web_data = {
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": rec[['match_id', 'league', 'home_team', 'away_team', 'Action', 'prob_Fav', 'Pat_L']].to_dict(orient='records')
            }
            with open(FINAL_JSON, 'w', encoding='utf-8') as f:
                json.dump(web_data, f, ensure_ascii=False, indent=4)
            print(f"🎉 全流程完成！結果已存至 {FINAL_JSON}")
    except Exception as e:
        print(f"❌ 預測階段出錯: {e}")

if __name__ == "__main__":
    main()
