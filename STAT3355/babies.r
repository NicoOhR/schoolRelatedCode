library(UsingR)
library(ggplot2)

data(fat)
ggplot(data = fat) +
  geom_point(mapping = aes(y = neck, x = wrist), color = "darkorange") + 
  labs(x = "Wrist circumference (cm)", y = "Neck circumference",
       title = "Neck Vs. Wrist Circumference") + 
  geom_abline(mapping = aes(intercept = 0, slope = 2), linetype = 2) +
  geom_abline(mapping = aes(intercept = a, slope = b), linetype = 1,
              color = "darkgreen") + 
  geom_hline(mapping = aes(yintercept = mean(neck)),
             color = "red", linetype = 2) +
  geom_vline(mapping = aes(xintercept = mean(wrist)),
             mmmcolor = "red", linetype = 2)
  

y <- fat$neck
x <- fat$wrist

y_bar <- mean(y)
x_bar <- mean(x)

n <- length(y)

b <-  (sum(x*y) - n*x_bar*y_bar)/(sum(x^2) - n*x_bar^2)

a <- y_bar - b*x_bar