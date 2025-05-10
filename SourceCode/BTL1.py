
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup, Comment
import pandas as pd
from io import StringIO
import time
from functools import reduce

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

required_columns = {
    'Player': 'Player',
    'Nation_stats_standard': 'Nation', 'Squad_stats_standard': 'Team',
    'Position_stats_standard': 'Position', 'Age_stats_standard': 'Age',
    'MP_stats_standard': 'Matches Played', 'Starts_stats_standard': 'Starts',
    'Min_stats_standard': 'Minutes', 'Gls_stats_standard': 'Goals',
    'Ast_stats_standard': 'Assists', 'CrdY_stats_standard': 'Yellow Cards',
    'CrdR_stats_standard': 'Red Cards', 'xG_stats_shooting': 'xG',
    'xAG_stats_passing_types': 'xAG', 'PrgC_stats_possession': 'PrgC',
    'PrgP_stats_possession': 'PrgP', 'PrgR_stats_possession': 'PrgR',
    'Gls_per90_stats_standard': 'Gls/90', 'Ast_per90_stats_standard': 'Ast/90',
    'xG_per90_stats_shooting': 'xG/90', 'xAG_per90_stats_passing_types': 'xAG/90',
    'GA90_stats_misc': 'GA90', 'Save%_stats_misc': 'Save%', 'CS%_stats_misc': 'CS%',
    'SoT%_stats_shooting': 'SoT%', 'SoT/90_stats_shooting': 'SoT/90',
    'G/Sh_stats_shooting': 'G/Sh', 'Dist_stats_shooting': 'Dist',
    'Cmp_stats_passing': 'Passes Completed', 'Cmp%_stats_passing': 'Cmp%',
    'TotDist_stats_passing': 'Pass Distance', 'Cmp%_stats_passing_types.1': 'Cmp% Short',
    'Cmp%_stats_passing_types.2': 'Cmp% Medium', 'Cmp%_stats_passing_types.3': 'Cmp% Long',
    'KP_stats_passing_types': 'Key Passes', '1/3_stats_passing_types': 'Pass Final 1/3',
    'PPA_stats_passing_types': 'Pass Pen Area', 'CrsPA_stats_passing_types': 'Cross to Pen Area',
    'PrgP_stats_passing_types': 'PrgP (passing)', 'SCA_stats_gca': 'SCA',
    'SCA90_stats_gca': 'SCA90', 'GCA_stats_gca': 'GCA', 'GCA90_stats_gca': 'GCA90',
    'Tkl_stats_defense': 'Tackles', 'TklW_stats_defense': 'Tackles Won',
    'Att_stats_defense': 'Challenges', 'Lost_stats_defense': 'Challenges Lost',
    'Blocks_stats_defense': 'Blocks', 'Sh_stats_defense': 'Shot Blocks',
    'Pass_stats_defense': 'Pass Blocks', 'Int_stats_defense': 'Interceptions',
    'Touches_stats_possession': 'Touches', 'Def Pen_stats_possession': 'Touches Def Pen',
    'Def 3rd_stats_possession': 'Touches Def 3rd', 'Mid 3rd_stats_possession': 'Touches Mid 3rd',
    'Att 3rd_stats_possession': 'Touches Att 3rd', 'Att Pen_stats_possession': 'Touches Att Pen',
    'Att_stats_possession': 'Take-ons Attempted', 'Succ%_stats_possession': 'Take-ons Success%',
    'Tkld%_stats_possession': 'Tkld%', 'Carries_stats_possession': 'Carries',
    'TotDist_stats_possession': 'Carry Distance', 'CPA_stats_possession': 'Carry to Pen Area',
    '1/3_stats_possession': 'Carry to 1/3', 'Mis_stats_possession': 'Miscontrols',
    'Dis_stats_possession': 'Dispossessed', 'Rec_stats_possession': 'Passes Received',
    'PrgR_stats_possession': 'PrgR', 'Fls_stats_misc': 'Fouls Committed',
    'Fld_stats_misc': 'Fouled', 'Off_stats_misc': 'Offsides',
    'Crs_stats_misc': 'Crosses', 'Recov_stats_misc': 'Recoveries',
    'Won_stats_misc': 'Aerials Won', 'Lost_stats_misc': 'Aerials Lost',
    'Won%_stats_misc': 'Aerial Win%'
}

table_links = {
    "stats_standard": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "stats_shooting": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
    "stats_passing": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
    "stats_passing_types": "https://fbref.com/en/comps/9/passing_types/Premier-League-Stats",
    "stats_gca": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
    "stats_defense": "https://fbref.com/en/comps/9/defense/Premier-League-Stats",
    "stats_possession": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
    "stats_misc": "https://fbref.com/en/comps/9/misc/Premier-League-Stats"
}

dfs = []
for table_id, url in table_links.items():
    print(f"Loading {table_id}...")
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    table = soup.find("table", {"id": table_id})
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            c_soup = BeautifulSoup(c, "lxml")
            table = c_soup.find("table", {"id": table_id})
            if table:
                break
    if not table:
        print(f"Table {table_id} not found, skipping")
        continue
    df = pd.read_html(StringIO(str(table)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
    if 'Player' in df.columns:
        df = df[df['Player']!='Player']
    df = df.drop_duplicates(subset=['Player'])  
    df = df.rename(columns={col: col if col=="Player" else f"{col}_{table_id}" for col in df.columns})   
    use_cols = ['Player'] + [col for col in df.columns if col in required_columns and col!='Player']
    dfs.append(df[use_cols])

driver.quit()

full_df = reduce(lambda left, right: pd.merge(left, right, on="Player", how="left"), dfs)

min_col = [c for c in full_df.columns if "Min" in c and "standard" in c][0]
full_df[min_col] = full_df[min_col].astype(str).str.replace(",","").str.extract(r"(\d+)")[0].astype(float)
full_df = full_df[full_df[min_col]>90]

full_df['FirstName'] = full_df['Player'].apply(lambda x: x.split(' ')[0])
full_df = full_df.sort_values('FirstName').fillna('N/a')

final_cols = [col for col in required_columns if col in full_df.columns]
output_df = full_df[final_cols].rename(columns=required_columns)
output_df.to_csv('results.csv', index=False)
print("results.csv created with", output_df.shape[1], "columns.")
