# Load Relevant Packages
library(pacman) # package manager

# tidyverse loads core tidyverse packages  
# Microsoft365R works with Overdrive
# here is the path manager
# janitor expedite data exploration and cleaning
# skimr provides a quick way to summarize tibbles
# purrr for functional programming
p_load(tidyverse, here, janitor, skimr, purrr)

# sets wd to use for all future relative paths
here() 



## Load data and recode
c17_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2017_0012.tsv") 
c17_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2017_0810.tsv") 
c18_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2018_0012.tsv") 
c18_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2018_0810.tsv") 
c19_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2019_0012.tsv") 
c19_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2019_0810.tsv") 
c20_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2020_0012.tsv") 
c20_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2020_0810.tsv") 
c21_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2021_0012.tsv") 
c21_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2021_0810.tsv") 
c22_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2022_0012.tsv") 
c22_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2022_0810.tsv") 
c23_0012 <- read_tsv("vaping_project_data/original_all_core/original_core_2023_0012.tsv") 
c23_0810 <- read_tsv("vaping_project_data/original_all_core/original_core_2023_0810.tsv") 

# Inspecting Data
# List of data frame names
data_frames <- c("c17_0012", "c17_0810", "c18_0012", "c18_0810", "c19_0012", 
                 "c19_0810", "c20_0012", "c20_0810", "c21_0012", "c21_0810", 
                 "c22_0012", "c22_0810", "c23_0012", "c23_0810")

# Function to apply skim_without_charts to the first 10 columns
skim_first_10 <- function(df_name) {
  df <- get(df_name)
  df %>% select(1:10) %>% skim_without_charts()
}

# Apply the function to each data frame
results <- map(data_frames, skim_first_10)

# Name the results list for easier reference
names(results) <- data_frames

# Print results
results
