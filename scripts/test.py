import pandas as pd

excel_file = "./Copy of torchview_ops.xlsx"
df = pd.read_excel(excel_file, sheet_name=1)  # 0 for first sheet, 1 for second

print(df.head())
