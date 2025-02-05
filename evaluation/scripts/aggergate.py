import pandas as pd
import  os
from  pathlib import Path

res = []

dir_path = r'./1205/'
result = list(Path("./1205/").rglob("*.csv"))

#print(result)
#print(len(result))

output_str="file, precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,accuracy,mcc\n"

for file in result:
    print(file)
    dataframe =pd.read_csv(file,)
    print((dataframe))
    avg_df = dataframe[["precision_0", "recall_0", "f-score_0","precision_1", "recall_1", "f-score_1", "accuracy", "mcc"]].mean()
    result_list =avg_df.to_list()
    output_str = output_str + str(file)+ "," +",".join(str(v) for v in   result_list) +"\n"


with open("oversample-more.csv", "w") as f:
    f.write(output_str)
    f.close()
