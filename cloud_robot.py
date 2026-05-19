import argparse
import time
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', type=str, default="")
    args = parser.parse_args()

    # 🛠️ 你的 Vercel 網頁主網址
    target_url = "https://smartmoneygrading.vercel.app/"
    
    with sync_playwright() as p:
        print("🚀 [系統] 正在雲端啟動 Chromium 虛擬瀏覽器環境...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        match_id_clean = args.match_id.strip()
        
        if match_id_clean and match_id_clean != "" and match_id_clean != "${{ github.event.client_payload.match_id }}":
            # 🎯 【手動強行介入模式】
            final_url = f"{target_url}?forceId={match_id_clean}"
            print(f"🔥 [Discord 密令] 強制單場精確狙擊！正在前往帶參網址: {final_url}")
            page.goto(final_url)
            print("⏳ 正在保持瀏覽器存活 15 秒，等待網頁 fetch 完成...")
            time.sleep(15)
            print("✅ [成功] 單場狙擊指令執行結束。")
            
        else:
            # 📡 【全網自動巡邏洗盤模式】
            print(f"📡 [整點巡邏] 執行全網雷達掃描。正在前往主控台: {target_url}")
            page.goto(target_url)
            time.sleep(3)

            # 🛡️ 核心黑科技：STEP 1 雙重確認與自動重試矩陣
            success_step1 = False
            for retry in range(1, 4): # 最多嘗試 3 次
                print(f"🔘 [動作] 正在模擬點擊: [STEP 1] 抓取全網未開賽事... (第 {retry}/3 次嘗試)")
                step1_btn = page.locator("button:has-text('STEP 1')")
                step1_btn.click()
                
                print("⏳ [等待] 數據集加載與 bf.js 雲端注入中 (預計等待 6 秒)...")
                time.sleep(6)
                
                # 💡 智能辨識：如果成功，畫面上一定會浮現 [STEP 2] 按鈕
                step2_btn = page.locator("button:has-text('STEP 2')")
                if step2_btn.is_visible():
                    print("✅ [保護機制] 成功偵測到 [STEP 2] 按鈕已浮現！代表 bf.js 順利載入。")
                    success_step1 = True
                    break
                else:
                    print("⚠️ [保護機制] 警告：網頁未浮現 [STEP 2]，可能 bf.js 載入遭丟包。執行網頁重整...")
                    page.reload()
                    time.sleep(3)

            if not success_step1:
                print("❌ [嚴重錯誤] 連續重試 3 次皆無法順利抓取全網賽事，中斷任務以節省雲端算力。")
                browser.close()
                return

            # ─── 順利進入 STEP 2 ───
            print("🔘 [動作] 正在模擬點擊: [STEP 2] 更新監控雷達...")
            step2_btn = page.locator("button:has-text('STEP 2')")
            step2_btn.click()
            time.sleep(4)

            print("🔍 [分析] 正在掃描監控雷達中浮現的賽事面板...")
            match_rows = page.locator("button.match-row")
            match_count = match_rows.count()
            print(f"📊 戰報：全網共掃描到 {match_count} 場符合過濾條件的監控賽事。")

            for i in range(match_count):
                row = match_rows.nth(i)
                match_text = row.inner_text().replace('\n', ' ')
                print(f"⚡ [雷達出擊 {i+1}/{match_count}] -> {match_text}")
                
                row.click()
                time.sleep(1.8) # 稍微放慢速度至 1.8 秒，更安全防止 Discord 頻率限制
            
            print("⏳ 正在進行最後 5 秒的緩衝收尾...")
            time.sleep(5)
            print("✅ [成功] 全網雷達整點洗盘任務圓滿結束。")

        browser.close()

if __name__ == "__main__":
    main()
