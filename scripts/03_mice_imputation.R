# MICE Multiple Imputation for Missing Data
#
# Implements Multivariate Imputation by Chained Equations (MICE) with M=5 imputations
# as described in the manuscript (Section 2.3.2, lines 277-292).
#
# This addresses missing data properly using multiple imputation, enabling
# Rubin's rules for pooling estimates across imputed datasets.
#
# Author: Siyang Ni
# Date: 2025-11-05

# Load packages
library(pacman)
p_load(tidyverse, mice, here)

cat("======================================================================\n")
cat(" MICE MULTIPLE IMPUTATION\n")
cat("======================================================================\n\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

data_path <- "~/work/vaping_project_data/processed_data_g12n.csv"

if (!file.exists(data_path)) {
  stop("ERROR: Data file not found at: ", data_path)
}

df <- read_csv(data_path, show_col_types = FALSE)
cat(sprintf("Data loaded: %d rows, %d columns\n", nrow(df), ncol(df)))

# ============================================================================
# 2. IDENTIFY VARIABLES FOR IMPUTATION
# ============================================================================

cat("\n======================================================================\n")
cat(" CHECKING MISSING DATA PATTERNS\n")
cat("======================================================================\n\n")

# Calculate missingness
missing_summary <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "N_Missing") %>%
  mutate(Pct_Missing = 100 * N_Missing / nrow(df)) %>%
  filter(N_Missing > 0) %>%
  arrange(desc(Pct_Missing))

cat("Variables with missing data:\n")
print(missing_summary, n = 50)

cat(sprintf("\nTotal variables with missing data: %d\n", nrow(missing_summary)))

# Identify variables to impute (those with >0% and <10% missing)
vars_to_impute <- missing_summary %>%
  filter(Pct_Missing > 0 & Pct_Missing < 10) %>%
  pull(Variable)

cat(sprintf("\nVariables to impute (0%% < missing < 10%%): %d\n", length(vars_to_impute)))

# ============================================================================
# 3. CONFIGURE MICE
# ============================================================================

cat("\n======================================================================\n")
cat(" CONFIGURING MICE\n")
cat("======================================================================\n\n")

# MICE configuration
M <- 5          # Number of imputations (as per manuscript)
MAXIT <- 20     # Maximum iterations
SEED <- 42      # Random seed for reproducibility

cat("MICE Configuration:\n")
cat(sprintf("  Number of imputations (M): %d\n", M))
cat(sprintf("  Maximum iterations: %d\n", MAXIT))
cat(sprintf("  Random seed: %d\n", SEED))
cat("\n")

# Set imputation methods
# - pmm (predictive mean matching) for continuous variables
# - logreg (logistic regression) for binary variables
# - polyreg (polytomous regression) for categorical with >2 levels

# Initialize MICE to determine default methods
init_mice <- mice(df, maxit = 0, seed = SEED, print = FALSE)
methods <- init_mice$method

# Review and adjust methods if needed
cat("Imputation methods (first 20 variables with missing):\n")
methods_df <- data.frame(
  Variable = names(methods[methods != ""]),
  Method = methods[methods != ""]
)
print(head(methods_df, 20))

# ============================================================================
# 4. RUN MICE IMPUTATION
# ============================================================================

cat("\n======================================================================\n")
cat(" RUNNING MICE IMPUTATION\n")
cat("======================================================================\n\n")

cat("This may take several minutes...\n\n")

# Run MICE
set.seed(SEED)
start_time <- Sys.time()

imp <- mice(df,
            m = M,                    # 5 imputations
            maxit = MAXIT,            # 20 iterations
            seed = SEED,              # Reproducibility
            printFlag = TRUE)         # Show progress

end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat(sprintf("\nMICE completed in %.2f minutes\n", as.numeric(elapsed_time)))

# ============================================================================
# 5. CHECK CONVERGENCE
# ============================================================================

cat("\n======================================================================\n")
cat(" CHECKING CONVERGENCE\n")
cat("======================================================================\n\n")

# Create convergence plot
convergence_plot_path <- "figures/mice_convergence.png"
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

png(convergence_plot_path, width = 12, height = 8, units = "in", res = 300)
plot(imp, layout = c(2, 2))
dev.off()

cat(sprintf("Convergence plot saved: %s\n", convergence_plot_path))
cat("Review this plot to ensure chains have converged (stable traces)\n")

# ============================================================================
# 6. EXPORT IMPUTED DATASETS
# ============================================================================

cat("\n======================================================================\n")
cat(" EXPORTING IMPUTED DATASETS\n")
cat("======================================================================\n\n")

output_dir <- "~/work/vaping_project_data"

# Export as long format (all M datasets stacked)
imputed_long <- complete(imp, action = "long", include = FALSE)
long_path <- file.path(output_dir, "imputed_data_m5_long.csv")
write_csv(imputed_long, long_path)
cat(sprintf("Saved long format: %s\n", long_path))
cat(sprintf("  Dimensions: %d rows x %d columns\n", nrow(imputed_long), ncol(imputed_long)))

# Export each imputation as separate CSV
for (i in 1:M) {
  imputed_i <- complete(imp, action = i)
  path_i <- file.path(output_dir, sprintf("imputed_%d.csv", i))
  write_csv(imputed_i, path_i)
  cat(sprintf("Saved imputation %d: %s\n", i, path_i))
}

# Also save with survey weights if they exist
if ("survey_weight" %in% names(df) || "ARCHIVE_WT" %in% names(df)) {
  cat("\nNote: Survey weights preserved in imputed datasets\n")
}

# ============================================================================
# 7. DIAGNOSTICS
# ============================================================================

cat("\n======================================================================\n")
cat(" IMPUTATION DIAGNOSTICS\n")
cat("======================================================================\n\n")

# Check density plots for selected variables
vars_to_check <- vars_to_impute[1:min(4, length(vars_to_impute))]

if (length(vars_to_check) > 0) {
  density_plot_path <- "figures/mice_density_plots.png"

  png(density_plot_path, width = 12, height = 8, units = "in", res = 300)
  densityplot(imp, ~ get(vars_to_check[1]) + get(vars_to_check[2]))
  dev.off()

  cat(sprintf("Density plot saved: %s\n", density_plot_path))
  cat("Review to ensure imputed values (red) match observed values (blue)\n")
}

# Summary statistics comparison
cat("\nComparing observed vs imputed values (first imputed variable):\n")
if (length(vars_to_impute) > 0) {
  first_var <- vars_to_impute[1]

  # Observed values
  observed <- df[[first_var]][!is.na(df[[first_var]])]

  # Imputed values (from first imputation)
  imputed_1 <- complete(imp, 1)[[first_var]][is.na(df[[first_var]])]

  cat(sprintf("\n%s:\n", first_var))
  cat("  Observed - Mean:", round(mean(observed, na.rm = TRUE), 3),
      " SD:", round(sd(observed, na.rm = TRUE), 3), "\n")
  cat("  Imputed  - Mean:", round(mean(imputed_1, na.rm = TRUE), 3),
      " SD:", round(sd(imputed_1, na.rm = TRUE), 3), "\n")
}

# ============================================================================
# 8. SAVE MICE OBJECT
# ============================================================================

cat("\n======================================================================\n")
cat(" SAVING MICE OBJECT\n")
cat("======================================================================\n\n")

mice_object_path <- file.path(output_dir, "mice_imputation_object.rds")
saveRDS(imp, mice_object_path)
cat(sprintf("MICE object saved: %s\n", mice_object_path))
cat("This can be loaded in R for further analysis using: readRDS()\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n======================================================================\n")
cat(" MICE IMPUTATION COMPLETE\n")
cat("======================================================================\n\n")

cat("Summary:\n")
cat(sprintf("  - %d variables had missing data\n", nrow(missing_summary)))
cat(sprintf("  - %d imputations created\n", M))
cat(sprintf("  - %d iterations per imputation\n", MAXIT))
cat(sprintf("  - Time elapsed: %.2f minutes\n", as.numeric(elapsed_time)))
cat("\nOutput files:\n")
cat("  - imputed_data_m5_long.csv (all imputations stacked)\n")
cat("  - imputed_1.csv through imputed_5.csv (separate files)\n")
cat("  - mice_imputation_object.rds (for R)\n")
cat("  - mice_convergence.png (diagnostic plot)\n")
cat("  - mice_density_plots.png (diagnostic plot)\n")
cat("\nNext steps:\n")
cat("  1. Review convergence plot to verify chains converged\n")
cat("  2. Review density plots to ensure imputed values are plausible\n")
cat("  3. Use imputed datasets in regression analysis\n")
cat("  4. Pool results using Rubin's rules (see 11_regression_with_mice.py)\n")
