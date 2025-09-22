data = read.csv("./hw2_systolic_bp(1).csv")
model <- lm(weight ~ sys.bp, data = data)
m <- summary(model)
m
#Want to test H_0 : \beta_1 = -50, first we calculate the test statistic
beta_1_hat <- coef(model)[2]
Sxx <- sum((data$weight - mean(data$weight))^2)
sigma_hat <- m$sigma
test_statistic <- (beta_1_hat - (-50))/sqrt(sigma_hat^2/Sxx)

#then we can calculate value of the t distribution 
n <- nrow(data)
t_critical <- qt(0.975, n -2)

