library(UsingR)

data("rivers")

x_bar <- mean(rivers)
s <- sd(rivers)
s2 <- IQR(rivers)
s3 <- mad(rivers)

z <- (rivers - x_bar)/s

sum(z < 3 & z > -3)/length(z)

barplot(z)
