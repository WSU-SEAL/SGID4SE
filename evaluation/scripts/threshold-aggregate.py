import  pandas as pd

import matplotlib.pyplot as plt
import  numpy as np

dataframe =pd.read_csv("results/vary-evaluation-GRU-oversample-random-ratio-1.333-bias-1-wc-False.csv")

dataframe['threshold'] =dataframe['threshold'].apply( lambda x: round(x, 2))
aggregateDF=dataframe.groupby("threshold").mean()

print( aggregateDF)

aggregateDF.to_excel("GRU-vary.xlsx")

plt.plot(aggregateDF["precision_1"], label ="Precision SGID", linestyle="-.")
plt.plot(aggregateDF["recall_1"], label ="Recall SGID", linestyle="--")
plt.plot(aggregateDF["f-score_1"], label ="F1-score SGID", linestyle=":")
plt.plot(aggregateDF["mcc"], label ="MCC", linestyle="-")



xmax = np.argmax(aggregateDF["mcc"])
xmax=(float)((xmax+1)/100)
y_max=max(aggregateDF["mcc"])
print(y_max)
print(xmax)
#print(aggregateDF["f-score_1"].shape)
plt.legend()
#plt.title("")

text = "Thrreshold={:.2f}, MCC={:.3f}".format(xmax, y_max)
#if not ax:
    #ax = plt.gca()
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
kw = dict(xycoords='data', textcoords="axes fraction",
          arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
plt.annotate(text, xy=(xmax, y_max), xytext=(0.6, 0.9), **kw)
plt.xlabel('SBERT' )
#plt.ylabel('metric')

#plt.annotate('max',xy=(xmax,y_max))
plt.show()
