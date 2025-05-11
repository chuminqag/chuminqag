from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import time

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)
wait = WebDriverWait(driver, 15)

driver.get('https://www.footballtransfers.com/en/values/players/most-valuable-players')

btn = wait.until(EC.element_to_be_clickable((
    By.CSS_SELECTOR,
    "button[data-target='#filterCompetitions']"
)))
btn.click()

item = wait.until(EC.element_to_be_clickable((
    By.XPATH,
    "//label[contains(., 'Premier League (UK)')]"
)))
item.click()

apply_btn = driver.find_element(
    By.CSS_SELECTOR,
    "button.js-filter-apply"   # thường là class js-filter-apply
)
apply_btn.click()

time.sleep(2)  
tbl = wait.until(EC.presence_of_element_located((
    By.CSS_SELECTOR,
    "table.table"  
)))

html = tbl.get_attribute('outerHTML')
df_values = pd.read_html(html)[0]

df_values = df_values[['Player', 'Value']]

def parse_val(txt):
    txt = txt.strip().lstrip('€')
    if txt.endswith('m'):
        return float(txt[:-1]) * 1_000_000
    if txt.endswith('k'):
        return float(txt[:-1]) * 1_000
    return float(txt.replace(',', ''))

df_values['TransferValue'] = df_values['Value'].astype(str).map(parse_val)

df_players = pd.read_csv('results.csv')
df_players = df_players[df_players['Minutes'] > 900].copy()
df_merged = pd.merge(df_players, df_values[['Player','TransferValue']],
                     on='Player', how='left')

df_merged.to_csv('transfer_values.csv', index=False, encoding='utf-8-sig')
print("✅ Đã lưu transfer_values.csv với giá chuyển nhượng Premier League.")

driver.quit()
