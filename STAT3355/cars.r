library(UsingR)
library(ggplot2)

data(mpg)

cty <- mpg$cty
hwy <- mpg$hwy

mpg$cmb <- (cty + hwy)/2


index_front <- which(mpg$drv == "f")
index_back <- which(mpg$drv == "4")

abs(median(mpg$cmb[index_front]) - median(mpg$cmb[index_back]))

index_t <- which(mpg$manufacturer == "toyota")

sd(mpg$cmb[index_t])

names(which.max(table(mpg$cyl)))

