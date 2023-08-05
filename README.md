# Supply_Chain

Introduction : Supply chain management (SCM) is the centralized management of the flow of goods and services and includes all processes that transform raw materials into final products.Management of supply chain includes storage capacities,facilities,location of warehouse and so on.
Current project aims at reducing the mismatch between supply and demand of a 'Noodles Company'.

Dataset : There are 20 features in the dataset(live data) and to get an in-depth intuition about these features market research Phase is recommended.
weight of the product(noodles) is the target variable.Hence it is a Regression Problem.

Data pre-processing has been performed to take of missing values and Outlier treatment is done in feature engineering phase.

EDA : Exploratory data analysis has been performed to establish the relationships between independent and target variable and all insights have been verified by performing hypothesis testing with each independent against the target variable.Hypothesis tests like Pearson correlation test for continuos variables against target variable. Two sample t-test and one way anova test has been performed for categorical variables against target variable.

Feature enginnering Phase: Feature engineering has been performed by creating new features from existing ones to improve the performance of the model and again these features has been analysed for their impact on target by hypothesis testing and appropriate insights.
Multicollinearity test has been performed using VIF but their is no significant relationship exists between independent variables.

Model-building: More than eight models have been tested and XGBoost Regressor is the best performing model with 99.4% accuracy.
Evaluation metrics like R2 Score, MSE, MAE are used. All metrics have been visualized using learning Curves(along with cross validation)

Feature Importance with XGBoost has been performed and recommendations for best features is given.
