# Use to get working dir in RStudio:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# To install ppcor::
install.packages("ppcor")

#Should really refit labels if we do this:
df1 <- read.csv("entropy_analysis.csv")
df2 <- read.csv("entropy_analysis_dataset.csv")
df <- rbind(df1, df2)


# Load csv
df <- read.csv("entropy_analysis_coil100TEST.csv")
df <- subset(df, accuracy < 0.99)  # Remove outliers

# Reduced models to get individual R squared:
model_full <- lm(accuracy ~ spectral_entropy + semantic_density, data = df)
model_reduced_entropy <- lm(accuracy ~ semantic_density, data = df)
model_reduced_density <- lm(accuracy ~ spectral_entropy, data = df)

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




# Pairwise Correlations (non-partial!)
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

















