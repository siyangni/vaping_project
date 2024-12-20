# Load Relevant Packages
library(pacman) # package manager

# tidyverse loads core tidyverse packages  
# Microsoft365R works with Overdrive
# here is the path manager
# janitor expedite data exploration and cleaning
# skimr provides a quick way to summarize tibbles
# purrr for functional programming
p_load(tidyverse, here, janitor, skimr, purrr, caret, gplots, pheatmap)



# Preprocessing 

## nicotine vaping


# Define a function to perform the transformation
transform_nicotine <- function(df, old_var) {
  df %>%
    mutate(nicotine12 = .data[[old_var]]) %>%
    select(-all_of(old_var))
}

# List of dataframes and corresponding old variable names
waves <- list(
  c17_0012 = "V2568", c17_0810 = "V7650",
  c18_0012 = "V2568", c18_0810 = "V7650",
  c19_0012 = "V2581", c19_0810 = "V7762",
  c20_0012 = "V2581", c20_0810 = "V7762",
  c21_0012 = "V7781", c21_0810 = "V7781",
  c22_0012 = "V7781", c22_0810 = "V7781",
  c23_0012 = "V7781", c23_0810 = "V7781"
)

# Apply the transformation to each dataframe in the list
for (wave in names(waves)) {
  assign(wave, transform_nicotine(get(wave), waves[[wave]]))
}

### Check recoding
tibble_names <- c("c17_0012", "c17_0810", "c18_0012", "c18_0810", "c19_0012",
                  "c19_0810", "c20_0012", "c20_0810", "c21_0012", "c21_0810",
                  "c22_0012", "c22_0810", "c23_0012", "c23_0810")

#### Function to calculate frequency and percentage of nicotine12
calculate_frequency <- function(tibble_name) {
  tibble <- get(tibble_name)
  tibble %>%
    count(nicotine12, drop = FALSE) %>%
    mutate(Percentage = n / sum(n) * 100) %>%
    mutate(tibble = tibble_name) %>%
    select(tibble, everything())  #### Ensure tibble name is the first column
}

#### Use purrr::map to apply the function to each tibble
frequency_list <- tibble_names %>%
  map(calculate_frequency)

#### Combine all results into a single data frame
frequency_df <- bind_rows(frequency_list)

#### View the result
print(frequency_df, n=83)



## Grade

### Function to add "grade" column for tibbles ending with "0810"
add_grade_column <- function(tibble_name) {
  if (grepl("0810$", tibble_name)) {
    tibble <- get(tibble_name)  # Get the tibble by name
    tibble <- tibble %>%
      mutate(grade = V501) %>%   # Add the 'grade' column
      select(-V501)  # Remove the 'V501' column
    assign(tibble_name, tibble, envir = .GlobalEnv)  # Assign the modified tibble back to the global environment
  }
}

### Apply the function to each tibble in the list
walk(tibble_names, add_grade_column)

### For 12th grade

### Function to add "grade" column with value 12 for tibbles ending with "0012"
add_grade_column_0012 <- function(tibble_name) {
  if (grepl("0012$", tibble_name)) {
    tibble <- get(tibble_name)  # Get the tibble by name
    tibble <- tibble %>%
      mutate(grade = 12)  # Add the 'grade' column with value 12
    assign(tibble_name, tibble, envir = .GlobalEnv)  # Assign the modified tibble back to the global environment
  }
}

### Apply the functions to each tibble in the list
walk(tibble_names, add_grade_column_0012)

### Check 
c19_0012 %>% 
  count(grade)

c20_0810 %>% 
  count(grade)

c23_0012 %>% 
  count(grade)

c23_0810 %>% 
  count(grade)

## COhort Year

### Function to add "wave" column based on tibble name
add_wave_column <- function(tibble_name) {
  # Extract the year from the tibble name
  year <- as.numeric(substring(tibble_name, 2, 3)) + 2000
  
  # Get the tibble by name
  tibble <- get(tibble_name)
  
  # Add the 'wave' column with the extracted year
  tibble <- tibble %>%
    mutate(wave = year)
  
  # Assign the modified tibble back to the global environment
  assign(tibble_name, tibble, envir = .GlobalEnv)
}

# Apply the function to each tibble in the list
walk(tibble_names, add_wave_column)

# Verify the changes
# Check 
c19_0012 %>% 
  count(wave)

c23_0012 %>% 
  count(wave)


## Sex

### Recode male as 0, and 1 as female, all others as NAs

# Funvtion to add the sex column
add_sex_column <- function(df, col_name) {
  df %>%
    mutate(sex = case_when(
      .data[[col_name]] == 1 ~ 0,
      .data[[col_name]] == 2 ~ 1,
      TRUE ~ NA_real_
    )) %>%
    select(-.data[[col_name]])  # Remove the original column
}

## For tibbles with V2150
v2150_tibbles <- list(c17_0012, c18_0012, c19_0012, c20_0012, c21_0012, c22_0012, c23_0012)
v2150_names <- paste0("c", 17:23, "_0012")

v2150_result <- map2(v2150_tibbles, v2150_names, ~add_sex_column(.x, "V2150")) %>%
  set_names(v2150_names)

## For tibbles with V7202
v7202_tibbles <- list(c17_0810, c18_0810, c19_0810, c20_0810, c21_0810, c22_0810, c23_0810)
v7202_names <- paste0("c", 17:23, "_0810")

v7202_result <- map2(v7202_tibbles, v7202_names, ~add_sex_column(.x, "V7202")) %>%
  set_names(v7202_names)

## Assign results back to original variable names
list2env(v2150_result, .GlobalEnv)
list2env(v7202_result, .GlobalEnv)

# Check
c20_0012 %>% 
  count(sex)

c23_0810 %>% 
  count(sex)


## Race
# Create a new variable race, copy either V2151 or V1070

## Helper function to add race column
add_race_column <- function(df_name) {
  df <- get(df_name, envir = .GlobalEnv)
  
  if (endsWith(df_name, "0012")) {
    df <- df %>%
      mutate(race = V2151) %>%
      select(-V2151)  # Remove the original column
  } else if (endsWith(df_name, "0810")) {
    df <- df %>%
      mutate(race = V1070) %>%
      select(-V1070)  # Remove the original column
  }
  
  assign(df_name, df, envir = .GlobalEnv)
}

## Apply the function to all tibbles
walk(tibble_names, add_race_column)

# Check
c17_0810 %>% 
  count(race)

c23_0012 %>% 
  count(race)



# Dichotomize the label

# Function to add nicotine12d
add_nicotine12d <- function(df) {
  df %>%
    mutate(nicotine12d = case_when(
      nicotine12 %in% c(-9, -8) ~ NA_real_,  # Missing data
      nicotine12 == 1 ~ 0,                   # Non-smoker
      nicotine12 >= 2 ~ 1,                   # Smoker
      TRUE ~ NA_real_                        # Catch-all for any unexpected values
    ))
}

# Loop through each dataset and add nicotine12d
for (name in tibble_names) {
  # Check if the dataset exists in the environment
  if (exists(name)) {
    # Retrieve the dataset
    df <- get(name)
    
    # Add the nicotine12d variable
    df <- add_nicotine12d(df)
    
    # Assign the modified dataframe back to the original name
    assign(name, df)
    
    # Optional: Print a message indicating completion
    message(paste("Processed dataset:", name))
  } else {
    warning(paste("Dataset", name, "does not exist in the environment."))
  }
}

# Verify
c23_0012 %>% 
  count(nicotine12d)



# Merge all 12th grade data

# Identify the common columns across all datasets
common_columns <- Reduce(intersect, list(
  colnames(c17_0012),
  colnames(c18_0012),
  colnames(c19_0012),
  colnames(c20_0012),
  colnames(c21_0012),
  colnames(c22_0012)
))

# Retain only the common columns in each dataset
datasets <- list(c17_0012, c18_0012, c19_0012, c20_0012, c21_0012, c22_0012)
datasets_common <- lapply(datasets, function(df) df[, common_columns])

# Merge all datasets with a full outer join (retain all rows)
merged_data <- Reduce(function(x, y) full_join(x, y, by = common_columns), datasets_common)

# Save the merged dataset to a CSV file
write.csv(merged_data, file = "vaping_project_data/merged_data_g12.csv", row.names = FALSE)




######################## Preparing  #########################################
rm(list = ls())

new_data <- read.csv("vaping_project_data/merged_data_g12.csv")

colnames(new_data)



# Remove all redudant dichotimize drug-use predictors
# Define the columns to remove
cols_to_remove <- c(
  # all redundant dichotomized drug use 
  "V2101D", "V2102D", "V2104D", "V2105D", "V2106D",
  "V2115D", "V2116D", "V2117D", "V2118D", "V2119D",
  "V2120D", "V2121D", "V2122D", "V2123D", "V2127D",
  "V2128D", "V2129D", "V2133D", "V2134D", "V2135D",
  "V2136D", "V2137D", "V2138D", "V2145D", "V2146D",
  "V2147D", "V2142D", "V2143D", "V2144D",
  # Useless info
  "RESPONDENT_ID", "V1", "V3", "ARCHIVE_WT", "V16", "V17", "V2190",
  # original label
  "nicotine12",
  # 0 variance
  "grade",
  # Remove auto-correlation
  "V2102", "V2104", "V2020", "V7963", "V7966", "V2104", "V2106",
  "V2020", "V2022", "V7957", "V7959", "V2115", "V2117", "V2118",
  "V2120", "V2121", "V2123", "V2032", "V2034", "V2124", "V2126",
  "V2459", "V2461", "V2042", "V2044", "V2127", "V2129", "V2029",
  "V2031", "V2133", "V2135", "V2136", "V2138", "V2139", "V2141",
  "V2142", "V2144", "V2145", "V2147", "V2493", "V2495", "V7783",
  "V7785", "V7786", "V7788", "V7724",
  # Remove highly-correlated
  "V2167", "V2170", "V2174", "V2192", "V2042", "V2103","V2107",
  "V2043",
  # Remove vape nicotine related
  "V2566", "V7780", "V7782", "V7789", "V7791", "V7793", "V7884",
  "V7887",
  # Remove high-missingness-percentage column (above 70%)
  "V2205", "V2206", "V2207", "V2200", "V2199", "V2198", 
  "V2204", "V2203", "V2202", "V2305", "V2918", "V2576", 
  "V2549", "V2564", "V2548", "V2547", "V2927", "V2009",
  "V2912", "V2003", "V2920", "V2307", "V2909", "V2919",
  "V2021"
)

# Remove the specified columns
new_data <- new_data %>% select(-any_of(cols_to_remove))
colnames(new_data)


# Check for missing values
sum(is.na(new_data))

# Identify Numerical Columns and Count Negative Values
negative_counts_df <- new_data %>%
  select_if(is.numeric) %>%                # Select numerical columns
  summarise_all(~ sum(. < 0, na.rm = TRUE)) %>%  # Count negatives in each column
  pivot_longer(cols = everything(), 
               names_to = "Column", 
               values_to = "Negative_Count")     # Convert to long format

## Display the Results
print(negative_counts_df, n = 129)

# Define the negative values that represent missing data
missing_codes <- c(-9, -8)

# Recode these values to NA in all numeric columns
new_data <- new_data %>%
  mutate(across(where(is.numeric), ~ ifelse(. %in% missing_codes, NA, .)))

# Number of missing values per column
missing_counts <- sapply(new_data, function(x) sum(is.na(x)))

# Percentage of missing values per column
missing_percent <- sapply(new_data, function(x) round(mean(is.na(x)) * 100, 2))

# Combine into a data frame for clarity
missing_summary <- data.frame(
  Missing_Count = missing_counts,
  Missing_Percentage = missing_percent
)

# Arrange the summary in descending order of missingness
missing_summary <- missing_summary[order(-missing_summary$Missing_Percentage), ]

# Display the summary
print(missing_summary)



### Correlation Analysis

## Check Variable type
str(new_data)

# Select numerical variables excluding the target variable if desired
# Assuming 'nicotine12d' is the target variable
cor_vars <- new_data %>%
  select(-nicotine12d) %>%
  select_if(is.numeric)

# Compute Spearman correlation matrix
# Use the spearman method because the features are not normally distributed and contains ordinal variables
cor_matrix_spearman <- cor(cor_vars, use = "pairwise.complete.obs", method = "spearman")

# View the Spearman correlation matrix
print(cor_matrix_spearman)

# Enhanced heatmap with clustering, color gradient, and annotations
pheatmap(cor_matrix_spearman, 
         color = colorRampPalette(c("blue", "white", "red"))(50),
         border_color = NA,                # Remove borders between tiles
         display_numbers = FALSE,           # Show correlation coefficients
         number_format = "%.2f",           # Format for numbers
         fontsize_number = 8,              # Font size for numbers
         clustering_method = "complete",    # Clustering method
         show_rownames = TRUE, 
         show_colnames = TRUE,
         main = "Enhanced Spearman Correlation Heatmap")


# Find highly correlated pairs (absolute correlation > 0.7)
high_corr_pairs <- which(abs(cor_matrix_spearman) > 0.5 & abs(cor_matrix_spearman) < 1, arr.ind = TRUE)

# Convert to a data frame
high_corr_df <- data.frame(
  Variable1 = rownames(cor_matrix_spearman)[high_corr_pairs[, 1]],
  Variable2 = colnames(cor_matrix_spearman)[high_corr_pairs[, 2]],
  Correlation = cor_matrix_spearman[high_corr_pairs]
)

# Remove duplicate pairs
high_corr_df <- high_corr_df[high_corr_df$Variable1 < high_corr_df$Variable2, ]

# View highly correlated pairs
print(high_corr_df)

# Correlation analysis show autocorrelation between drug use variables, all of them are removed
# Also remove are highly correlated pairs

# Save the preprocessed dataset to a CSV file
write.csv(new_data, file = "vaping_project_data/processed_data_g12.csv", row.names = FALSE)






