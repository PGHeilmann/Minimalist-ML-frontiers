# Stacked ensembles on basis of parentage information can predict hybrid performance with an accuracy comparable to marker-based GBLUP

## File S3: Code example

Example procedure to conduct a Gradient Boosting Machine grid search
and form Stacked Ensembles on the basis of the results using h2o
and a corn dataset from Technow et al. 2014. Procedure as described in Heilmann et al. 2023.


```{r}
# Load required packages
library(sommer)
library(h2o)

# Start the h2o cluster
h2o.init()

# Load the dataset
data("DT_technow")

# Categorical variables are required to be of type 'factor'
DT_technow$dent   <- as.factor(DT_technow$dent)
DT_technow$flint  <- as.factor(DT_technow$flint)
DT_technow$hy     <- as.factor(DT_technow$hy)

### Hyperparameter search space for grid search
# All of these hyperparameter levels will be considered in the random selection
# of hyperparameter combinations used in the random grid search.
# Can be extended or modified  

params <- list(ntrees      = seq(200,1000,200),     # Number of trees to include
               max_depth   = seq(2, 20, 2),         # Max. depth of each tree
               min_rows    = c(1,3,5,8,10),         # Min. rows required for split
               sample_rate = seq(0.1, 1.0, 0.1),    # Fraction of rows sampled per tree
               nbins_cats  = NA)                    # gets replaced later

search_criteria <- list(strategy   = "RandomDiscrete", # Set for random grid search 
                        max_models = 10,               # Train 50 models
                        seed       = 2102)             # Seed for reproducibility

# Creates list 'idx.list' with indices for 100 random splits
# Split ratio was 90% training set to 10% test set
# Splits can be generated again by setting the seed and sending
# the command again. This is to ensure somewhat reproducible results.

seed.vec <- 2017
idx.list <- list()
set.seed(seed.vec)

# Each iteration creates a random split, stores it in a list
for (x in 1:100) {                                            
  idx.list[[x]] <- sample(x    = 1:nrow(DT_technow),
                          size = round(nrow(DT_technow)*0.9))  
}


# Seeds for the random processes within the ML algorithm
# Not to be confused with seed set for choosing
# the hyperparameters.
# Setting this seed does not provide 100% reproducibility
# but results are closer to each other
set.seed(123123)
lseed <- sample(1:10000, 100)

gbm_cors  <- c()     # To store gbm prediction accuracy
ens_cors  <- c()     # To store ensemble prediction accuracy
grid_list <- list()  # To store grid details

for (i in 1:100) {
  # Get the indicies for the training set in this iteration
  idx   <- idx.list[[i]]
  
  # Assign training set by index
  Train <- DT_technow[idx, ]
  
  # Hence, removing the training set indices leaves the test set
  Test  <- DT_technow[-idx, ]
  
  # Check if any parental line are contained in the test set but not the
  # training set. If such is the case, remove them from the test set
  # Both parental lines need to be available in the training set for GCA
  # to work
  
  if (!all(Test$dent %in% Train$dent)) {
    Test <- Test[Test$dent %in% Train$dent, ]
  }
  if (!all(Test$flint %in% Train$flint)) {
    Test <- Test[Test$flint %in% Train$flint, ]
  }
  
  # Dataframes need to be transformed into h2o.frames for further use
  Train.h2o <- as.h2o(Train[c("GY", "flint", "dent")])
  Test.h2o  <- as.h2o( Test[c("GY", "flint", "dent")])
  
  # We set the nbins_cats hyperparameter space according
  # to the number of factor levels, i.e. parental lines, in our 
  # training set. Depending on the train/test split, this may change
  n_cat <- length(levels(Train$flint)) + length(levels(Train$dent))
  
  # Modify the hyperparameter search space
  # We use 100%, 50%, 25% or 5% of factor levels
  params$nbins_cats <- round(c(n_cat, n_cat * .5, n_cat * .25, n_cat * .05))
  
  # Run the grid search
  # Depending on the computational capacity and threads used, this may take a while
  # Using the standard metric 'mean residual deviance' is identical to MSE for this task
  gbm_grid <-
    h2o.grid( algorithm       =  "gbm",                   # Algorithm for grid search
              y               = "GY",                     # Target variable
              x               = c("flint", "dent"),       # Predictor variables
              training_frame  = Train.h2o,                # Data, must be in h2o format
              grid_id         = paste0("gbm_grid_", i),   # Name of the grid
              nfolds          = 10,                       # 10-fold CV 
              seed            = lseed[i],                 # seed for random procedures 
              hyper_params    = params,                   # predefined hyperparameters
              search_criteria = search_criteria,          # predefined criteria
              keep_cross_validation_predictions = TRUE    # save CV - required for SE
    )   
  
  # Get best grid search model
  best_model <- h2o.getModel(gbm_grid@model_ids[[1]])
  
  # Make predictions with best model, transform h2o.frane to dataframe
  gbm_preds <- as.data.frame(h2o.predict(best_model, Test.h2o))[, ]
  
  # Store accuracy in vector 
  gbm_cors  <- c(gbm_cors, cor(Test$GY, gbm_preds, method = "pearson"))
  
  # Visualize predicted compared to observed yield
  plot(
    Test$GY,
    gbm_preds,
    pch = 20,
    col = "black",
    xlim = c(min(Test$GY),   max(Test$GY)),
    ylim = c(min(gbm_preds), max(gbm_preds))
  )
  
  # Store grid details and ranking in a list
  grid_list[[i]] <- as.data.frame(gbm_grid@summary_table)
  
  
  ### This finds the optimum number of models for the
  ### Stacked Ensemble and predicts yield using the best
  
  # Create dataframe to store number of models
  # and prediction accuracy
  ensemble_eval <- data.frame()
  
  # Iterate over sequence of Top 5, 10, ..., 50 models
  for (x in seq(5,50,5)) {
    ensemble <-
      h2o.stackedEnsemble(
        y                     = "GY",                                # target 
        x                     = c("flint", "dent"),                  # predictors 
        metalearner_algorithm = "glm",                               # super learner
        metalearner_params    = list(lambda_search = T, alpha = 0),  # Ridge Regression 
        metalearner_nfolds    = 10,                                  # 10-fold CV 
        training_frame        = Train.h2o,                           # in h2o format
        base_models           = c(unlist(gbm_grid@model_ids[1:x]))   # model names
      )
    
    # Save the r2 as a measure of prediction accuracy here
    r2 <- ensemble@model$metalearner_model@model$cross_validation_metrics@metrics$r2
    r2 < as.vector(r2)
    
    # Store no. of models included and accuracy in dataframe
    ensemble_eval = rbind(ensemble_eval,c(x,r2) )
  }
  
  # Choose no. of models with highest accuracy
  ens_model_num <- ensemble_eval[which.max(ensemble_eval[,2]),1]
  # n_models <- c(n_models,ens_model_num)
  
  # Build Stacked Ensemble with best no. of models
  ensemble <-
    h2o.stackedEnsemble(
      y                     = "GY",                                
      x                     = c("flint", "dent"),                  
      metalearner_algorithm = "glm",                               
      metalearner_params    = list(lambda_search = T, alpha = 0),   
      metalearner_nfolds    = 10,                                  
      training_frame        = Train.h2o,                           
      base_models           = c(unlist(gbm_grid@model_ids[1:ens_model_num]))  
    )
  
  # Make predictions with best ensemble, transform h2o.frane to dataframe 
  ens_preds <- as.data.frame(h2o.predict(ensemble, Test.h2o))[, ]

  # Store accuracy in vector 
  ens_cors <- c(ens_cors, cor(Test$GY, ens_preds, method = "pearson"))
}
```
