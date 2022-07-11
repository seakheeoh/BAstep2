library(SHAPforxgboost)
library(DALEX)

#SHAP summary plot
shap_values <- shap.values(xgb_model = model, X_train = train.x)
shap_long <- shap.prep(xgb_model = model, X_train = train.x)
shap.plot.summary(shap_long)

#Partial dependent plot
fig_list <- lapply(names(shap_values$mean_shap_score)[1:4], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 2)

#SHAP force plot
shap_values <- shap.values(xgb_model = model, X_train = test.X)
plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 7, n_groups = 1)
shap.plot.force_plot(plot_data, zoom_in_location = c(0,200), y_parent_limit = c(-3,8))
shap.plot.force_plot_bygroup(plot_data)




