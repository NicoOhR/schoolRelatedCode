library(UsingR)

data("samhda")

str(samhda)

unique(samhda$gender)


index_7 <- which(samhda$gender == 7)
samhda$gender[index_7] <- NA
samhda$gender <- factor(samhda$gender, levels = 1:2, labels = c("M", "F"))

table(samhda$alcohol)
index_9 <- which(samhda$alcohol == 9)
samhda$alcohol[index_9] <- NA
samhda$alcohol <- samhda$alcohol == 1

table(samhda$marijuana)
index_9 <- which(samhda$marijuana == 9)
samhda$marijuana[index_9] <- NA
samhda$marijuana <- samhda$marijuana == 1

table(samhda$gender, samhda$alcohol)
F <- xtabs(~ gender + alcohol, data = samhda)

table(samhda$marijuana, samhda$alcohol)

F <- xtabs(~ marijuana + alcohol, data = samhda)
P <- F/sum(F)
