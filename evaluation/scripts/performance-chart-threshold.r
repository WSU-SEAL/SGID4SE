library(ggplot2)
library(dplyr)
library(reshape2)


dataset <- read.csv(file = file.choose(), header = TRUE)


dataset = dataset[-1,]
print(dataset)
df<-dataset[, c("Algorithm", "Threshold", "Precision", "Recall", "F1.Score", "MCC")] 


plotdata <- melt(df,id = c("Algorithm", "Threshold"), value.name="measure")


dat_text <- data.frame(
  label = c("Threshold: 0.28, MCC: 0.767", 
            "Threshold: 0.74, MCC: 0.751", 
            "Threshold: 0.70, MCC: 0.758",
            "Threshold: 0.99, MCC: 0.739", 
            "Threshold: 0.94, MCC: 0.806", 
            "Threshold: 0.84, MCC: 0.769"),
  Algorithm   = c("CNN", "GRU","LSTM", "ALBERT", "BERT", "SBERT"),
  threshold     = c(0.28, 0.74, 0.7, 0.99, 0.94, 0.84),
  variable     = c("MCC", "MCC", "MCC", "MCC", "MCC", "MCC"),
  measure = c(0.767, 0.751,  0.758, 0.739, 0.806, 0.769)
)




ggplot(plotdata, aes(x=Threshold, y=measure, group=variable, color=variable)) + 
  geom_line()+ xlab("Threshold") + 
  ylab("Performance")+    labs(colour = "Metrics") +  theme(legend.position="top")+  
  facet_wrap(~ Algorithm, ncol = 3) + 
  geom_text(data    = dat_text, color="black",  show.legend=FALSE,   mapping = aes(x = -Inf, y = -Inf,   label = label,  hjust   = -0.15,  vjust   = -1))+
geom_vline(data = data.frame(xint=0.28,Algorithm="CNN"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
 geom_vline(data = data.frame(xint=0.74,Algorithm="GRU"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
  geom_vline(data = data.frame(xint=0.70,Algorithm="LSTM"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
  geom_vline(data = data.frame(xint=0.99,Algorithm="ALBERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")+
  geom_vline(data = data.frame(xint=0.94,Algorithm="BERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")+
  geom_vline(data = data.frame(xint=0.84,Algorithm="SBERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")


