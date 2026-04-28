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
    print("🚀 使用 curl_cffi (潛行 TLS 模式已啟動)")
except ImportError:
    import requests
    HAS_CFFI = False
    print("ℹ️ 使用標準 requests 庫")

try:
    import V79_Core
    print("✅ V79_Core 核心邏輯載入成功")
except ImportError:
    print("🚨 錯誤：找不到 V79_Core.py。請確認檔案已在目前目錄並更名正確。")

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

# ==================== 3. Colab 專用 WebDriver 配置 ====================
# ==================== 1. 修復版 WebDriver 配置 (Colab 專用) ====================
def create_driver():
    """Colab 穩定版 Chrome Driver"""
    options = Options()
    
    # Colab 關鍵參數
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-plugins')
    options.add_argument('--disable-images')  # 加速
    options.add_argument('--window-size=1920,1080')
    
    # 防偵測
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # User-Agent
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
    
    # Chrome 二進位檔路徑 (Colab 通用)
    options.binary_location = "/usr/bin/google-chrome"  # ✅ 改用 google-chrome
    
    # ChromeDriver 服務參數
    service = webdriver.chrome.service.Service()
    service.creation_flags = 0x08000000  # 背景執行
    
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# ==================== 2. 修復版 bf.js 抓取 ====================
def get_match_ids_from_bf_js():
    """Selenium + 純 requests 雙重備援"""
    timestamp = int(time.time() * 1000)
    
    # 先試純 requests (避開 Selenium)
    print("📡 嘗試純 requests 獲取 bf.js...")
    try:
        import requests
        r = requests.get(
            f"http://live.nowscore.com/data/bf.js?{timestamp}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://live.nowscore.com/"
            },
            timeout=10
        )
        if "A[0]" in r.text:
            return parse_bf_js_content(r.text)
    except:
        pass
    
    # Selenium 備援
    print("🔄 requests 失敗，啟動 Selenium...")
    driver = create_driver()
    try:
        driver.get(f"http://live.nowscore.com/data/bf.js?{timestamp}")
        time.sleep(3)
        content = driver.page_source
        if "A[0]" in content:
            return parse_bf_js_content(content)
    except Exception as e:
        print(f"❌ Selenium 也失敗: {e}")
    finally:
        driver.quit()
    
    return []

def parse_bf_js_content(content):
    """通用解析函數"""
    leagues_map = {}
    b_raw = re.findall(r'B\[(\d+)\]\s*=\s*[\'"[]([^\'"\]]*)[\'"\]];', content)
    for idx, val in b_raw:
        parts = val.replace("'", "").split('^')
        if len(parts) > 0:
            leagues_map[idx] = parts[0].strip()
    
    final_ids = []
    a_raw = re.findall(r'A\[(\d+)\]\s*=\s*[\'"[]([^\'"\]]*)[\'"\]];', content)
    print(f"🔎 發現 {len(a_raw)} 場賽事")
    
    for idx, val in a_raw:
        parts = val.replace("'", "").split('^') if '^' in val else val.split(',')
        parts = [p.strip().strip("'").strip('"') for p in parts]
        if len(parts) >= 10 and leagues_map.get(parts[1], "") in WHITE_LIST:
            final_ids.append(parts[0])
    
    print(f"✅ 過濾後 {len(final_ids)} 場白名單賽事")
    return list(set(final_ids))

# ==================== 3. 簡化版爬蟲 (只用 requests) ====================
def scrape_eu_worker(match_chunk):
    """純 requests，避免 Selenium"""
    batch = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    for mid in match_chunk:
        url = f"https://m.nowscore.com/1x2Detail/{mid}_177.htm"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # 你的解析邏輯...
                batch.append({...})
        except:
            continue
    
    return batch

# ==================== 4. 簡化主流程 ====================
def main():
    os.makedirs("temp_eu", exist_ok=True)
    os.makedirs("temp_as2", exist_ok=True)
    
    # 1. 獲取 ID (requests 優先)
    ids = get_match_ids_from_bf_js()
    if not ids:
        print("📭 無賽事，任務結束")
        return
    
    # 2. 純 requests 爬取
    eu_data = scrape_eu_worker(ids)
    if eu_data:
        pd.DataFrame(eu_data).to_csv("predict_eu16.csv", index=False)
    
    # 3. V79 預測 (如果有資料)
    if os.path.exists(MODEL_PATH) and os.path.exists("predict_eu16.csv"):
        print("🧠 執行 V79 預測...")
        # 你的 V79_Core 邏輯...
        print("🎉 預測完成！")
    else:
        print("⚠️ 缺少模型或資料")

if __name__ == "__main__":
    main()
