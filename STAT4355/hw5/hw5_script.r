df <- read.csv("football.csv")
op <- par(no.readonly = TRUE)
on.exit(par(op), add = TRUE)
par(mfrow = c(3, 3), mar = c(3.5, 3.5, 2, 1))

for (j in paste0("x", 1:9)) {
  plot(df[[j]], df[["y"]],
       xlab = j, ylab = "Games won (y)",
       main = paste(j, "vs y"),
       pch = 20, cex = 0.8)
}

fit <- lm("y ~ x1 + x2 + x5 + x7 + x8", data=df)
summary(fit)

anova(fit)

fit0<-lm(y~1, data=df)
# anova(fit0, fit)
