# Use to get working dir in RStudio:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# To install ppcor::
install.packages("ppcor")

### RUN START ###
### To run the analysis first compute <1, then <0.65

df <- read.csv("entropy_analysis.csv")
df <- subset(df, accuracy < 0.65)  # Remove outliers

# Reduced models to get individual R squared:
model_full <- lm(accuracy ~ spectral_entropy + semantic_density, data = df)
model_reduced_entropy <- lm(accuracy ~ semantic_density, data = df)
model_reduced_density <- lm(accuracy ~ spectral_entropy, data = df)

summary(model_full)

# Get R^2 Values:
r2_full <- summary(model_full)$r.squared
r2_reduced_entropy <- summary(model_reduced_entropy)$r.squared
r2_reduced_density <- summary(model_reduced_density)$r.squared

# Compute Cohens f^2 for each variable
f2_entropy <- (r2_full - r2_reduced_entropy) / (1 - r2_full)
f2_density <- (r2_full - r2_reduced_density) / (1 - r2_full)

# Print
cat("Cohen's f^2 for spectral_entropy:", round(f2_entropy, 3), "\n")
cat("Cohen's f^2 for semantic_density:", round(f2_density, 3), "\n")



# Extract residuals
res <- residuals(model_full)

# Print first few residuals
print("Residuals:")
print(head(res))

# Shapiro-Wilk Normality Test
print("Shapiro-Wilk Normality Test:")
shapiro.test(res)


# Plots of residuals
# Q-Q plot 
print("Q-Q Plot:")
qqnorm(res)
qqline(res, col = "red")

# Hist over residuals
hist(res,
     breaks = 20,
     col = "lightblue",
     main = "Histogram of Model Residuals",
     xlab = "Residuals",
     ylab = "Frequency",
     border = "black")


# lowess point cloud plot
plot(df$spectral_entropy, df$semantic_density)
lines(lowess(df$spectral_entropy, df$semantic_density), col = "blue")



# Pairwise Correlations (non-partial!) not particularly useful here
print("Correlations")
cor.test(df$accuracy, df$spectral_entropy)
cor.test(df$accuracy, df$semantic_density)
cor.test(df$spectral_entropy, df$semantic_density)



##################################
### Partial Correlation Tests: ###
##################################


# Load package:
library(ppcor)

# pcor.test is correlation between X,Y controlling for Z in .test(X,Y,Z)
# Order of X,Y doesn't matter!


# Partial Correlation controlling for density:
pcor_result <- pcor.test(df$accuracy, df$spectral_entropy, df$semantic_density)
print(pcor_result)

# Partial Correlation controlling for Spectral Entropy:
pcor_result <- pcor.test(df$accuracy, df$semantic_density, df$spectral_entropy)
print(pcor_result)

# Partial Correlation controlling for Spectral Entropy:
pcor_result <- pcor.test(df$semantic_density, df$spectral_entropy, df$accuracy)
print(pcor_result)

# Ranked tests which you should use if the residuals aren't normal!

# Partial Correlation controlling for density:
pcor_result <- pcor.test(df$accuracy, df$spectral_entropy, df$semantic_density, method = "spearman")
print(pcor_result)

# Partial Correlation controlling for Spectral Entropy:
pcor_result <- pcor.test(df$accuracy, df$semantic_density, df$spectral_entropy,  method = "spearman")
print(pcor_result)

# Partial Correlation controlling for Spectral Entropy:
pcor_result <- pcor.test(df$semantic_density, df$spectral_entropy, df$accuracy,  method = "spearman")
print(pcor_result)


### ------------------------------------------- ###

# Small test for difference in Spectral Entropies
# Figures come from Eploratory plotting notebook
processed <- c(
  0.7097, 0.6841, 0.5421, 0.5387, 0.5178, 0.5080, 0.4675, 0.4308, 0.4288,
  0.3807, 0.3736, 0.2472, 0.2294, 0.1810, 0.1747, 0.0732, 0.0352
)

non_processed <- c(
  0.7088, 0.6837, 0.5609, 0.5605, 0.5109, 0.2571, 0.5460, 0.2852, 0.3604,
  0.4321, 0.3381, 0.1975, 0.2562, 0.1923, 0.1485, 0.0849, 0.0405
)

diffs <- processed - non_processed

# 3. Normality test
shapiro.test(diffs)

# We reject normality
# Plots for good measure, clearly we lack enough points to draw conclusions.

hist(diffs, main = "Histogram of Differences",
     xlab = "Processed - Non-processed", col = "skyblue", border = "white")

qqnorm(diffs)
qqline(diffs, col = "red", lwd = 2)

# 6. Paired Wilcoxon signed-rank test
wilcox.test(processed, non_processed, paired = TRUE)

# No difference in proportion p-value = 0.6777


### CIFAR TEST


df <- read.csv("cifartest.csv")
model <- lm(accuracy ~ semantic_density, data = df)

r2 = summary(model)$r.squared
f2 = r2 / (1-r2)
f2


# Extract residuals
res <- residuals(model)

# Print first few residuals
print("Residuals:")
print(head(res))

# Shapiro-Wilk Normality Test
print("Shapiro-Wilk Normality Test:")
shapiro.test(res)


# Plots of residuals
# Q-Q plot 
print("Q-Q Plot:")
qqnorm(res)
qqline(res, col = "red")


model <- lm(accuracy ~ (semantic_density), data = df)
res <- residuals(model)
shapiro.test(res)
print("Q-Q Plot:")
qqnorm(res)
qqline(res, col = "red")

hist(res,
     breaks = 20,
     col = "lightblue",
     main = "Histogram of Model Residuals",
     xlab = "Residuals",
     ylab = "Frequency",
     border = "black")


cor.test(df$accuracy,df$semantic_density, method = "spearman")

mean(df$accuracy)
mean(df$semantic_density)






##################################
### Par. Corr. Confidence Int. ###
##################################

bootstrap_pcor <- function(x, y, z, data, method = "spearman", n = 1000, conf = 0.95) {
  library(ppcor)
  
  set.seed(123) 
  
  estimates <- replicate(n, {
    idx <- sample(1:nrow(data), replace = TRUE)
    df_sample <- data[idx, ]
    result <- pcor.test(df_sample[[x]], df_sample[[y]], df_sample[[z]], method = method)
    return(result$estimate)
  })
  
  alpha <- 1 - conf
  ci <- quantile(estimates, probs = c(alpha / 2, 1 - alpha / 2))
  estimate <- median(estimates)
  
  return(list(
    estimate = round(estimate, 3),
    conf.int = round(ci, 3)
  ))
}


df <- read.csv("entropy_analysis.csv")
df <- subset(df, accuracy < 0.65)  #<- 1/0.65

# 1. Accuracy ~ Spectral Entropy | Semantic Density
result_SE <- bootstrap_pcor("accuracy", "spectral_entropy", "semantic_density", data = df, method = "pearson")

# 2. Accuracy ~ Semantic Density | Spectral Entropy
result_SD <- bootstrap_pcor("accuracy", "semantic_density", "spectral_entropy", data = df, method = "pearson")

# 3. Spectral Entropy ~ Semantic Density | Accuracy
result_SE_SD <- bootstrap_pcor("spectral_entropy", "semantic_density", "accuracy", data = df, method = "pearson")

print("Partial Correlation: Accuracy ~ SE | SD")
print(result_SE$conf.int)

print("Partial Correlation: Accuracy ~ SD | SE")
print(result_SD$conf.int)

print("Partial Correlation: SE ~ SD | Accuracy")
print(result_SE_SD$conf.int)

# NO conditioning <- use SPEARMAN METHOD
# rho_1 = -0.237  0.101
# rho_2 = -0.964 -0.897 

# CONDITIONED <- PEARSON METHOD
# rho_1 = -0.551 -0.063
# rho_2 = -0.763 -0.205 

########### CIFAR ############

bootstrap_spearman_ci <- function(x, y, data, n = 1000, conf = 0.95) {
  set.seed(123)
  rho_vals <- replicate(n, {
    idx <- sample(1:nrow(data), replace = TRUE)
    df_sample <- data[idx, ]
    cor(df_sample[[x]], df_sample[[y]], method = "spearman")
  })
  
  alpha <- 1 - conf
  ci <- quantile(rho_vals, probs = c(alpha / 2, 1 - alpha / 2))
  estimate <- median(rho_vals)
  
  return(list(
    estimate = round(estimate, 3),
    conf.int = round(ci, 3)
  ))
}

df <- read.csv("cifartest.csv")
model <- lm(accuracy ~ semantic_density, data = df)

result <- bootstrap_spearman_ci("accuracy", "semantic_density", df)
print(result)

# rho = -0.806 -0.590 <- CIFAR test CI










