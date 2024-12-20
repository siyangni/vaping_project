# Load Relevant Packages
library(pacman) # package manager

# tidyverse loads core tidyverse packages  
# Microsoft365R works with Overdrive
# here is the path manager
# janitor expedite data exploration and cleaning
# skimr provides a quick way to summarize tibbles
# purrr for functional programming
p_load(tidyverse, here, janitor, skimr, purrr)
p_load(caret) # for ML
p_load(glmnet) # For logistic model
p_load(randomForest) # For random forest algorithm
p_load(doParallel) # Parallel Processing
p_load(MLmetrics) # For model metrics

# Clear RAM
rm(list = ls())

# Load the CSV file into a new data frame
new_data <- read.csv("merged_data_g12.csv")

# Check the structure of the loaded dataset
str(new_data)
# Print the first few rows to verify
head(new_data)



# Define variables to exclude
excluded_vars <- c("RESPONDENT_ID", "V1", "V3", "ARCHIVE_WT", "V13", "V16", "V17", "wave",
                   "V2101D", "V2102D", "V2104D", "V2105D", "V2106D", "V2115D", "V2116D", "V2117D",        
                   "V2118D", "V2119D", "V2120D", "V2121D", "V2122D", "V2123D",        
                   "V2127D", "V2128D",  "V2129D", "V2133D", "V2134D", "V2135D",        
                   "V2136D", "V2137D",  "V2138D", "V2145D", "V2146D", "V2147D",        
                   "V2142D", "V2143D",  "V2144D")

# Subset data to include only the required predictors and target variable
predictors <- setdiff(names(new_data), c("nicotine12", excluded_vars))
subseted_data <- new_data[, c("nicotine12", predictors)]


# Split the dataset into training and testing sets
set.seed(666)
train_index <- caret::createDataPartition(subseted_data$nicotine12, p = 0.8, list = FALSE)
train_data <- subseted_data[train_index, ]
test_data <- subseted_data[-train_index, ]

# Create the cross-list deletion version of the data
train_data_nona <- na.omit(train_data)
test_data_nona <- na.omit(test_data)

# Prepare data
x_train <- model.matrix(nicotine12 ~ ., train_data_nona)[, -1]
y_train <- train_data_nona$nicotine12

# Set control parameters for cv.glmnet
set.seed(666)  # for reproducibility
nfolds <- 5    # reduced from default 10
nlambda <- 50  # reduced from default 100

# Setup parallel processing
cores <- detectCores() - 1
registerDoParallel(cores)

# Fit optimized multinomial logistic regression model
glmnet_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  nfolds = nfolds,
  nlambda = nlambda,
  parallel = TRUE,
  type.measure = "class"  # for classification accuracy
)

# Clean up parallel processing
stopImplicitCluster()

# Get coefficients at optimal lambda
coefficients <- coef(glmnet_model, s = "lambda.min")

# Print coefficients for each class
print(coefficients)

# Get variable importance
var_importance <- lapply(coefficients, function(x) {
  # Get absolute values of coefficients
  abs_coef <- abs(as.matrix(x))[-1, ]  # Remove intercept
  # Sort by importance
  sort(abs_coef, decreasing = TRUE)
})

# Print top 10 most important variables for each class
lapply(var_importance, head, 10)




##### Random Forest
# Function to sanitize class labels
sanitize_class_labels <- function(labels) {
  # Convert to character if not already
  labels <- as.character(labels)
  
  # Replace '-' with 'neg'
  labels <- gsub("-", "neg", labels)
  
  # Prefix with 'C' to ensure it doesn't start with a digit
  labels <- paste0("C", labels)
  
  # Ensure names are valid R variable names
  labels <- make.names(labels)
  
  return(labels)
}


# Prepare feature matrix and target variable for training and testing
x_train_rf <- train_data_nona %>% 
  select(-nicotine12)  # Exclude target variable

x_test_rf <- test_data_nona %>% 
  select(-nicotine12)  # Exclude target variable

y_train_rf <- as.factor(train_data_nona$nicotine12)  # Convert to factor for classification
y_test_rf <- as.factor(test_data_nona$nicotine12)  # Convert to factor for classification

# Apply to training labels
original_levels_train <- levels(y_train_rf)
sanitized_levels_train <- sanitize_class_labels(original_levels_train)

# Apply to testing labels
original_levels_test <- levels(y_test_rf)
sanitized_levels_test <- sanitize_class_labels(original_levels_test)

# Ensure that both train and test have the same mapping
if(!identical(sanitized_levels_train, sanitized_levels_test)) {
  stop("Sanitized class labels for training and testing data do not match. Please check the mappings.")
}

# Update the factor levels with sanitized labels
levels(y_train_rf) <- sanitized_levels_train
levels(y_test_rf) <- sanitized_levels_test

# Verify the new levels
print(unique(y_train_rf))
# Expected Output: "Cneg9" "C1" "C2" "C3" "C4" "C5" "C6" "C7"

print(unique(y_test_rf))
# Expected Output: "Cneg9" "C1" "C2" "C3" "C4" "C5" "C6" "C7"


# Define cross-validation method
train_control <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary  # For multiclass classification
)

# Register parallel backend if not already registered
# (You have already registered it for glmnet, so this step is optional)
# library(doParallel)
# registerDoParallel(cores)
# Setup parallel processing
cores <- detectCores() - 1
registerDoParallel(cores)


# Set seed for reproducibility
set.seed(666)

# Train the Random Forest model
rf_model <- train(
  x = x_train_rf,
  y = y_train_rf,
  method = "rf",
  trControl = train_control,
  metric = "Accuracy",  # Optimize for classification accuracy
  tuneLength = 5,       # Number of tuning parameters to try
  importance = TRUE      # Enable variable importance
)

# Clean up parallel processing
stopImplicitCluster()

# View the results of hyperparameter tuning
print(rf_model)
plot(rf_model)

# Make predictions on the train set
rf_predictions <- predict(rf_model, newdata = x_train_rf)

# For probabilistic predictions (useful for ROC)
rf_prob_predictions <- predict(rf_model, newdata = x_train_rf, type = "prob")



