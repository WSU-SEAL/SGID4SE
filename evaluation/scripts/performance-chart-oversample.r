library(ggplot2)
library(dplyr)
library(reshape2)

#ratio/strategy variation
dataset <- read.csv(file = file.choose(), header = TRUE)

df<-dataset[, c("Algorithm", "Ratio", "MCC", "Strategy")] 


ggplot(df, aes(x=Ratio, y=MCC, group=Strategy, color=Strategy)) + 
  geom_line()+ xlab("Sample ratio: # of GSD/ # of non-GSD") + 
  ylab("MCC")+    labs(colour = "Oversampling Strategy") +  theme(legend.position="top")+  
  facet_wrap(~ Algorithm, ncol = 5)  


#class bias variation


dataset <- read.csv(file = file.choose(), header = TRUE)

df<-dataset[, c("Algorithm", "Weight", "MCC", "Precision", "Recall", "F1.score" )] 




plotdata <- melt(df,id = c("Algorithm", "Weight"), value.name="measure")

print(df)


ggplot(plotdata, aes(x=Weight, y=measure, group=variable, color=variable)) + 
  geom_line()+ xlab("Minority class bias") + 
  ylab("Performance")+    labs(colour = "Metrics") +  theme(legend.position="top")+  
  facet_wrap(~ Algorithm, ncol = 4)  
