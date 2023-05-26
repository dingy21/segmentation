# Customer Profiling and Segmentation
To achieve the goal of getting better understanding of our customers and identifying who are most likely to respond positively to the next marketing campaign, this project will be mainly focused on customer profiling by using k-means cluster and predictive analysis by building logistic regression model.
## Load Library
```
library(C50)
library(tidyverse)
library(tidymodels)
library(janitor)
library(skimr)
library(kableExtra)
library(GGally)
library(kableExtra)
library(vip)
library(fastshap)
library(MASS)
library(rpart.plot)
library(factoextra)
library(imputeMissings)
library(ISLR)
library(tree)
```
## Import Data
```
mkt <- read_csv("marketing_campaign.csv") %>% clean_names()
new <- read_csv("new_customers_mkt.csv") %>% clean_names()

mkt = subset(mkt, select = -c(z_cost, z_rev))
```
## Exploratory Analysis
```
mkt_summary <- mkt %>%
  count(response) %>%
  mutate(pct = n/sum(n))

mkt_summary %>%
  ggplot(aes(x = factor(response), y = pct)) +
  geom_col() +
  geom_text(aes(label = round(pct, 2)), vjust = 2.5, color = "white") +
  labs(title = "Customers' Repsonse to Last Marketing Campaign", x = "Response", y = "Pct")
```
## Data Preparation for Clustering
```
mkt <- mkt %>%
  mutate(age = 2022 - birth)

# -- impute numeric with median
mkt1 <- mkt %>% imputeMissings::impute()

# -- scale the numeric data
mkt1$age <- scale(mkt1$age)
mkt1$income <- scale(mkt1$income)
mkt1$kids <- scale(mkt1$kids)
mkt1$wines <- scale(mkt$wines)
mkt1$fruits <- scale(mkt$fruits)
mkt1$meat <- scale(mkt$meat)
mkt1$fish <- scale(mkt$fish)
mkt1$sweets <- scale(mkt$sweets)
mkt1$gold <- scale(mkt$gold)
mkt1$web <- scale(mkt$web)
mkt1$deals <- scale(mkt1$deals)
mkt1$web <- scale(mkt1$web)
mkt1$catalog <- scale(mkt1$catalog)
mkt1$store <- scale(mkt1$store)

mkt1 = subset(mkt1, select = -c(id, dt_customer, mar_stat, birth, education, response, teens, recency, visits))

skim(mkt1)
```
## Visually Choose Number of Clusters
```
fviz_nbclust(mkt1, kmeans, method = "wss")
```
![Picture1](https://github.com/dingy21/segmentation/assets/134649288/3c795cea-f6d7-40db-af20-86ea9901fa4a)
## Generate and Visualize Clusters
```
set.seed(904)

cluster <- kmeans(mkt1, 7, iter.max = 200, nstart = 10)
print(cluster)

fviz_cluster(cluster, mkt1, ellipse.type = "norm", geom = "point")
```
![Picture2](https://github.com/dingy21/segmentation/assets/134649288/7762f379-ca13-435c-8b89-af4d427e60c9)
## Visualize Cluster Segmentation
```
ggplot(mkt1,aes(cluster$cluster))+geom_bar()

ggplot(mkt1,aes(x=age))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=age))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=income))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=income))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=wines))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=wines))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=fruits))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=fruits))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(meat))+geom_bar()
ggplot(mkt1,aes(meat))+geom_bar()+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(fish))+geom_bar()
ggplot(mkt1,aes(fish))+geom_bar()+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(gold))+geom_bar()
ggplot(mkt1,aes(gold))+geom_bar()+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=deals))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=deals))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=cmplain))+geom_histogram(binwidth=1)
ggplot(mkt1,aes(x=cmplain))+geom_histogram(binwidth=1)+facet_wrap(~cluster$cluster)

ggplot(mkt1,aes(x=cluster$cluster))+geom_bar()+facet_wrap(~mkt$response) + labs(title = "Response by Clusters")

ggplot(mkt,aes(response))+geom_bar()+facet_wrap(~cluster$cluster)+geom_text(stat='count', aes(label=..count..), vjust = 1.2, colour = "white")+labs(title = "Response by Clusters")
```
![Picture3](https://github.com/dingy21/segmentation/assets/134649288/eead2787-0f77-418f-aee1-e381f332539e)
## Data Preparation for Regression
```
mktfin <- mkt %>%
  mutate(cmp1 = as.factor(cmp1)) %>%
  mutate(cmp2 = as.factor(cmp2)) %>%
  mutate(cmp3 = as.factor(cmp3)) %>%
  mutate(cmp4 = as.factor(cmp4)) %>%
  mutate(cmp5 = as.factor(cmp5)) %>%
  mutate(education = as.factor(education)) %>%
  mutate(mar_stat = as.factor(mar_stat)) %>%
  mutate(cmplain = as.factor(cmplain)) %>%
  mutate(response = as.factor(response))

skim(mktfin)


new_pred <- new %>%
  mutate(cmp1 = as.factor(cmp1)) %>%
  mutate(cmp2 = as.factor(cmp2)) %>%
  mutate(cmp3 = as.factor(cmp3)) %>%
  mutate(cmp4 = as.factor(cmp4)) %>%
  mutate(cmp5 = as.factor(cmp5)) %>%
  mutate(education = as.factor(education)) %>%
  mutate(mar_stat = as.factor(mar_stat)) %>%
  mutate(cmplain = as.factor(cmplain))

skim(new_pred)
```
## Data Partition
```
set.seed(43)
mkt_split <- initial_split(mktfin, prop = 0.7)
train <- training(mkt_split)
test <- testing(mkt_split)

sprintf("Train PCT : %1.2f%%", nrow(train)/nrow(mktfin) * 100)
sprintf("Test  PCT : %1.2f%%", nrow(test)/nrow(mktfin) * 100)
```
## Recipe for Full Model
```
mkt_recipe <- recipe(response ~ ., data = train) %>%
  step_rm(id, birth, dt_customer) %>%
  step_impute_median(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  prep()

bake_train <- bake(mkt_recipe, new_data = train)
bake_test <- bake(mkt_recipe, new_data = test)
```
## Logistic Regression - Full Model
```
logistic_glm <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(response ~ ., data = bake_train)

tidy(logistic_glm) %>%
  mutate_at(c("estimate", "std.error", "statistic", "p.value"), round, 4)

logistic_glm %>%
  vi()
```
### Evaluation
```
# -- training 
predict(logistic_glm, bake_train, type = "prob") %>%
  bind_cols(.,predict(logistic_glm, bake_train)) %>%
  bind_cols(.,bake_train) -> scored_train_glm

head(scored_train_glm)

# -- testing 
predict(logistic_glm, bake_test, type = "prob") %>%
  bind_cols(.,predict(logistic_glm, bake_test)) %>%
  bind_cols(.,bake_test) -> scored_test_glm

head(scored_test_glm)

event_level = 'first'

# -- AUC: Train and Test 
scored_train_glm %>% 
  metrics(response, .pred_1, estimate = .pred_class) %>%
  mutate(part = "training") %>%
  bind_rows(scored_test_glm %>%
              metrics(response, .pred_1, estimate = .pred_class) %>%
              mutate(part = "testing"))

# -- ROC Charts 
scored_train_glm %>%
  mutate(model = "train") %>%
  bind_rows(scored_test_glm %>%
              mutate(model = "test")) %>%
  group_by(model) %>%
  roc_curve(response, .pred_1) %>%
  autoplot()

# -- Confusion Matrix  
scored_train_glm %>%
  conf_mat(response, .pred_class) %>%
  autoplot(type = "heatmap") +
  labs(title="Train Confusion Matrix")

scored_test_glm %>%
  conf_mat(response, .pred_class) %>%
  autoplot(type = "heatmap") +
  labs(title="Test Confusion Matrix")
```
![Picture4](https://github.com/dingy21/segmentation/assets/134649288/57f7224c-8200-4a20-bb73-02886f7b462c)
## Reduce Model Using Stepwise
```
steplog <- glm(response ~ ., data = bake_train, family = binomial(link = "logit"))
step <- stepAIC(steplog, direction = "both")
summary(step)
```
## Step Recipe
```
finalrecipe <- recipe(response ~ teens + recency + meat + deals + web + catalog + store + visits + education + mar_stat + cmp3 + cmp4 + cmp5 + cmp1 + cmp2, data = train) %>%
  step_impute_mode(all_nominal(), -all_outcomes()) %>%
  step_impute_median(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  prep()

finalrecipe
```
## Apply New Recipe to Final Model
```
bake_finaltrain <- bake(finalrecipe, new_data = train)
bake_finaltest <- bake(finalrecipe, new_data = test)

logis_final <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(response ~ ., data = bake_finaltrain)

tidy(logis_final) %>%
  mutate_at(c("estimate", "std.error", "statistic", "p.value"), round, 4)
```
## Prepare for Final Evaluation
```
predict(logis_final, bake_train, type = "prob") %>%
  bind_cols(., predict(logis_final, bake_train)) %>%
  bind_cols(., bake_train) -> scored_train_final

head(scored_train_final)


predict(logis_final, bake_test, type = "prob") %>%
  bind_cols(., predict(logis_final, bake_test)) %>%
  bind_cols(., bake_test) -> scored_test_final

head(scored_train_final)
```
## Final Model Evaluation
```
options(yardstick.event_first = FALSE)

# -- evaluation metrics
scored_train_final %>%
  metrics(response, .pred_1, estimate = .pred_class) %>%
  mutate(part = "training") %>%
  bind_rows(scored_test_final %>%
              metrics(response, .pred_1, estimate = .pred_class) %>%
              mutate(part = "testing"))

# -- auc_roc curve
scored_train_final %>%
  mutate(model = "train") %>%
  bind_rows(scored_test_final %>%
              mutate(model = "test")) %>%
  group_by(model) %>%
  roc_curve(response, .pred_1) %>%
  autoplot()

# -- confusion matrix
scored_train_final %>%
  conf_mat(response, .pred_class) %>%
  autoplot(type = "heatmap") +
  labs(title = "Train Confusion Matrix")

scored_test_final %>%
  conf_mat(response, .pred_class) %>%
  autoplot(type = "heatmap") +
  labs(title = "Test Confusion Matrix")
```
![Picture5](https://github.com/dingy21/segmentation/assets/134649288/efb94daf-f2b3-456d-b1ff-ef04a14ed73f)
