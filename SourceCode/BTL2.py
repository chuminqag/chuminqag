import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('results.csv')

# 2. Tiền xử lý: chuyển các cột object có thể sang numeric
for col in df.columns:
    if df[col].dtype == object:
        # Loại bỏ ký tự '%' và dấu phẩy, thay 'N/a' hoặc 'nan' thành None
        s = (df[col].astype(str)
               .str.replace('%', '', regex=False)
               .str.replace(',', '', regex=False)
             )
        s = s.replace({'N/a': None, 'nan': None})
        try:
            df[col] = s.astype(float)  # Chuyển sang float nếu được
        except ValueError:
            pass  # Giữ nguyên nếu không thể chuyển

# Lấy danh sách các cột thống kê (numeric)
stats = df.select_dtypes(include=['float', 'int']).columns.tolist()

# 3. Tìm Top 3 và Bottom 3 cầu thủ cho mỗi thống kê
with open('top_3.txt', 'w', encoding='utf-8') as f:
    for stat in stats:
        f.write(f'Statistic: {stat}\n')
        top3 = df.nlargest(3, stat)[['Player', stat]]
        bot3 = df.nsmallest(3, stat)[['Player', stat]]
        f.write('Top 3:\n')
        for _, row in top3.iterrows():
            f.write(f'  {row["Player"]}: {row[stat]}\n')
        f.write('Bottom 3:\n')
        for _, row in bot3.iterrows():
            f.write(f'  {row["Player"]}: {row[stat]}\n')
        f.write('\n')

# 4. Tính median, mean, std cho toàn giải và cho mỗi đội
records = []
for stat in stats:
    # Toàn giải
    median_all = df[stat].median()
    mean_all   = df[stat].mean()
    std_all    = df[stat].std()
    records.append({'Statistic': f'Median of {stat}', 'Scope': 'all', 'Value': median_all})
    records.append({'Statistic': f'Mean of {stat}',   'Scope': 'all', 'Value': mean_all})
    records.append({'Statistic': f'Std of {stat}',    'Scope': 'all', 'Value': std_all})
    # Từng đội
    for team, group in df.groupby('Team'):
        records.append({'Statistic': f'Median of {stat}', 'Scope': team, 'Value': group[stat].median()})
        records.append({'Statistic': f'Mean of {stat}',   'Scope': team, 'Value': group[stat].mean()})
        records.append({'Statistic': f'Std of {stat}',    'Scope': team, 'Value': group[stat].std()})

# Xuất kết quả ra CSV
results2 = pd.DataFrame(records)
results2.to_csv('results2.csv', index=False, encoding='utf-8-sig')
print("✅ Đã lưu results2.csv với", results2.shape[0], "dòng và", results2.shape[1], "cột.")

os.makedirs('histograms', exist_ok=True)
for stat in stats:
    # tạo tên file an toàn: chỉ giữ chữ, số, gạch dưới
    safe_stat = re.sub(r'[^0-9A-Za-z_]', '_', stat)

    # Histogram cho toàn giải
    plt.figure()
    df[stat].hist(bins=20)
    plt.title(f'Phân bố của {stat} (Tất cả cầu thủ)')
    plt.xlabel(stat)
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig(f'histograms/{safe_stat}_all.png')
    plt.close()

    # Histogram cho từng đội
    for team, group in df.groupby('Team'):
        safe_team = re.sub(r'[^0-9A-Za-z_]', '_', team)
        plt.figure()
        group[stat].hist(bins=20)
        plt.title(f'Phân bố {stat} ({team})')
        plt.xlabel(stat)
        plt.ylabel('Tần suất')
        plt.tight_layout()
        plt.savefig(f'histograms/{safe_stat}_{safe_team}.png')
        plt.close()


# 6. Xác định đội có mean cao nhất cho mỗi thống kê
best = {}
for stat in stats:
    team_best = df.groupby('Team')[stat].mean().idxmax()
    best[stat] = team_best

print("Đội xuất sắc nhất (theo mean) cho mỗi thống kê:")
for stat, team in best.items():
    print(f"{stat}: {team}")
