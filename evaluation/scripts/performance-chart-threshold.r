library(ggplot2)
library(dplyr)
library(reshape2)


dataset <- read.csv(file = file.choose(), header = TRUE)


dataset = dataset[-1,]
print(dataset)
df<-dataset[, c("Algorithm", "Threshold", "Precision", "Recall", "F1.Score", "MCC")] 


plotdata <- melt(df,id = c("Algorithm", "Threshold"), value.name="measure")


dat_text <- data.frame(
  label = c("Threshold: 0.53, MCC: 0.766", 
            "Threshold: 0.68, MCC: 0.738", 
            "Threshold: 0.52, MCC: 0.756",
            "Threshold: 0.54, MCC: 0.697", 
            "Threshold: 0.98, MCC: 0.807", 
            "Threshold: 0.84, MCC: 0.769"),
  Algorithm   = c("CNN", "GRU","LSTM", "ALBERT", "BERT", "SBERT"),
  threshold     = c(0.53, 0.68, 0.52, 0.54, 0.98, 0.84),
  variable     = c("MCC", "MCC", "MCC", "MCC", "MCC", "MCC"),
  measure = c(0.766, 0.738,  0.756, 0.697, 0.807, 0.769)
)




ggplot(plotdata, aes(x=Threshold, y=measure, group=variable, color=variable)) + 
  geom_line()+ xlab("Threshold") + 
  ylab("Performance")+    labs(colour = "Metrics") +  theme(legend.position="top")+  
  facet_wrap(~ Algorithm, ncol = 3) + 
  geom_text(data    = dat_text, color="black",  show.legend=FALSE,   mapping = aes(x = -Inf, y = -Inf,   label = label,  hjust   = -0.15,  vjust   = -1))+
geom_vline(data = data.frame(xint=0.53,Algorithm="CNN"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
 geom_vline(data = data.frame(xint=0.68,Algorithm="GRU"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
  geom_vline(data = data.frame(xint=0.52,Algorithm="LSTM"), aes(xintercept = xint), linetype = "longdash", colour = "blue") +
  geom_vline(data = data.frame(xint=0.54,Algorithm="ALBERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")+
  geom_vline(data = data.frame(xint=0.98,Algorithm="BERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")+
  geom_vline(data = data.frame(xint=0.84,Algorithm="SBERT"), aes(xintercept = xint), linetype = "longdash", colour = "blue")


