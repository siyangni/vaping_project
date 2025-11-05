# Load Relevant Packages
library(pacman) # package manager

# tidyverse loads core tidyverse packages
# Microsoft365R works with Overdrive
# here is the path manager
# janitor expedite data exploration and cleaning
# skimr provides a quick way to summarize tibbles
# purrr for functional programming
p_load(tidyverse, here, janitor, skimr, purrr, caret, gplots, pheatmap)


# Preprocessing WITH SURVEY WEIGHTS PRESERVED
# This is a modified version of 02_preprocessing.R that retains ARCHIVE_WT

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

## Wave

# Function to add wave number
add_wave_column <- function(tibble_name) {
  wave <- as.numeric(str_sub(tibble_name, 2, 3))  # Extract wave year
  tibble <- get(tibble_name)
  tibble <- tibble %>%
    mutate(wave = wave)
  assign(tibble_name, tibble, envir = .GlobalEnv)
}

walk(tibble_names, add_wave_column)

## Sex
add_sex_column <- function(df, var_name) {
  df %>%
    mutate(sex = .data[[var_name]]) %>%
    select(-all_of(var_name))
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

## Assign results back to global environment
for (i in seq_along(v2150_names)) {
  assign(v2150_names[i], v2150_result[[i]], envir = .GlobalEnv)
}

for (i in seq_along(v7202_names)) {
  assign(v7202_names[i], v7202_result[[i]], envir = .GlobalEnv)
}

## Race
# Create a new variable race, copy either V2151 or V1070

## Helper function to add race column
add_race_column <- function(df, df_name) {
  # Check if the df_name ends with "0012" or "0810" and apply transformation
  if (endsWith(df_name, "0012")) {
    df <- df %>%
      mutate(race = V2151) %>%
      select(-V2151)  # Remove the original column
  } else if (endsWith(df_name, "0810")) {
    df <- df %>%
      mutate(race = V1070) %>%
      select(-V1070)  # Remove the original column
  }

  return(df)
}

# Apply the function using map2
all_tibbles <- c(v2150_tibbles, v7202_tibbles)
all_names <- c(v2150_names, v7202_names)

# Use map2 to apply add_race_column
result_with_race <- map2(all_tibbles, all_names, add_race_column) %>%
  set_names(all_names)

# Assign the result back to global environment
for (i in seq_along(all_names)) {
  assign(all_names[i], result_with_race[[i]], envir = .GlobalEnv)
}

## Merge datasets

### Combine all tibbles into one
c17 <- bind_rows(c17_0012, c17_0810)
c18 <- bind_rows(c18_0012, c18_0810)
c19 <- bind_rows(c19_0012, c19_0810)
c20 <- bind_rows(c20_0012, c20_0810)
c21 <- bind_rows(c21_0012, c21_0810)
c22 <- bind_rows(c22_0012, c22_0810)
c23 <- bind_rows(c23_0012, c23_0810)

### Merge
all <- bind_rows(c17, c18, c19, c20, c21, c22, c23)

## Filter by grade 12

g12 <- all %>%
  filter(grade == 12)

# Recode nicotine vaping to binary
g12 <- g12 %>%
  mutate(nicotine12d = case_when(
    nicotine12 %in% c(1, 2, 3, 4, 5, 6) ~ 1,
    nicotine12 == 7 ~ 0,
    TRUE ~ NA_real_
  ))

# Check
table(g12$nicotine12, g12$nicotine12d, useNA = "ifany")

# MODIFIED: Columns to remove - ARCHIVE_WT is NO LONGER REMOVED
cols_to_remove <- c(
  # all redundant dichotomized drug use
  "V2101D", "V2102D", "V2104D", "V2105D", "V2106D",
  "V2115D", "V2116D", "V2117D", "V2118D", "V2119D",
  "V2120D", "V2121D", "V2122D", "V2123D", "V2127D",
  "V2128D", "V2129D", "V2133D", "V2134D", "V2135D",
  "V2136D", "V2137D", "V2138D", "V2145D", "V2146D",
  "V2147D", "V2142D", "V2143D", "V2144D",
  # Useless info (ARCHIVE_WT removed from this list)
  "RESPONDENT_ID", "V1", "V3", "V16", "V17", "V2190",
  # original label
  "nicotine12",
  # grade
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

# Remove columns
g12n <- g12 %>%
  select(-any_of(cols_to_remove))

# Rename ARCHIVE_WT to survey_weight for clarity
g12n <- g12n %>%
  rename(survey_weight = ARCHIVE_WT)

# Save with weights included
write_csv(g12n, "~/work/vaping_project_data/processed_data_g12n_with_weights.csv")

cat("\n=== Data processing complete with survey weights preserved ===\n")
cat("Output file: ~/work/vaping_project_data/processed_data_g12n_with_weights.csv\n")
cat("Survey weight column: survey_weight\n")
cat("Sample size:", nrow(g12n), "\n")
cat("Columns:", ncol(g12n), "\n")
