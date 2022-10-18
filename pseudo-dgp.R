library(mgcv)
library(scam)
library(rms)
library(tidyverse)

set.seed(2001)

x1 <- rnorm(10000)
x2 <- rnorm(10000)

# Population log odds is a function of sigmoid features

z <- -15 + 2 * SSfpl(x1, -2, 2, 0, 0.5) - 5 * SSfpl(x2, -1, 1, 0, 0.1) +
  rnorm(10000,0,5)

y <- rbinom(100000, 1, prob=boot::inv.logit(z))

full_data <- data.frame(y=y, x1=x1, x2=x2)

summary(full_data)

train_index <- caTools::sample.split(full_data$y, SplitRatio = 1/100)

train_data <- full_data[train_index,] # small
test_data <- full_data[!train_index,] # large

## fit a gam

## when population error is high, gam returns non-monotonic relationships
## even when the population relationships are monotonic

mod_gam <- gam(y ~ s(x1) + s(x2), data = train_data, family = binomial)

summary(mod_gam)

plot(mod_gam, pages=1, seWithMean=TRUE, main="Unconstrained GAM",
     shift=coef(mod_gam)[1])

## gam with monotonic constraints

## scam imposes monotonic relationships between each
## feature and the binary response
## mpi = monotonic p-spline increasing
## mpd = monotonic p-spline decreasing

mod_mono_gam <- scam(y ~ s(x1, bs="mpi") + s(x2, bs="mpd"),
                     data = train_data, family = binomial)

summary(mod_mono_gam)

plot(mod_mono_gam, pages=1, seWithMean=TRUE, main="Constrained GAM",
     shift=coef(mod_mono_gam)[1])

lapply(list(mod_gam, mod_mono_gam), AIC)

## test set performance

preds_gam <- predict(mod_gam, newdata=test_data, type="response")
preds_mono_gam <- predict(mod_mono_gam, newdata=test_data,
                          type="response")

lapply(list(gam=preds_gam, mono_gam=preds_mono_gam), function(x){
  caTools::colAUC(x, test_data$y)
})

lapply(list(gam=preds_gam, mono_gam=preds_mono_gam), function(x){
  MLmetrics::LogLoss(x, test_data$y)
})

## generate pseudo-data using Mono GAM

## suppose the scam model is the data generating process (dgp).
## given the dgp, we can simulate as much pseudo-training data as
## we want. in this case, we will simulate 250 copies of the training
## data. then use the scam probability predictions and rbinom to simulate
## binary outcomes for each observation -- for example: if scam
## predicts a 1% probability, we expect 2.5 copies of the observation
## to be "1" and 247.5 copies to be "0"

pseudo_train_data <- train_data %>% slice(rep(1:n(), each=250))

pseudo_train_data$pseudo_prob <- as.numeric(predict(mod_mono_gam,
                                                    newdata=pseudo_train_data,
                                         type="response"))

pseudo_train_data$y <- rbinom(n = nrow(pseudo_train_data),
                                 size = 1,
                                 prob = pseudo_train_data$pseudo_prob)

summary(pseudo_train_data)

## approximate the Mono GAM model with a restricted cubic spline model

## unfortunately, both gam and scam models are very difficult to 
## implement outside of R due to how the basis functions are defined.
## an alternative is to approximate the scam relationships
## with simpler basis functions like restricted cubic splines
## and the truncated power basis (which can be easily implemented
## in Excel)

ddist <- datadist(pseudo_train_data)
options(datadist = "ddist")

mod_logit_pseudo <- lrm(y ~ rcs(x1) + rcs(x2), data=pseudo_train_data)

ggplot(Predict(mod_logit_pseudo), addlayer=ggtitle("Pseudo Model"))

## express the model as truncated power basis
## this can implemented in Excel

score_function <- Function(mod_logit_pseudo)

score_function

## test set performance

preds_pseudo <- predict(mod_logit_pseudo, newdata=test_data,
                        type="fitted")

lapply(list(gam=preds_gam, mono_gam=preds_mono_gam, pseudo=preds_pseudo), function(x){
  caTools::colAUC(x, test_data$y)
})

lapply(list(gam=preds_gam, mono_gam=preds_mono_gam, pseudo=preds_pseudo), function(x){
  MLmetrics::LogLoss(x, test_data$y)
})

## Compare predict() against score_function

compare_df <- data.frame(x1 = seq(-5, 5, 1),
           x2 = seq(-5, 5, 1))

rms_preds <- predict(mod_logit_pseudo, newdata=compare_df, type="lp")

trunc_power_preds <- with(compare_df, score_function(x1=x1, x2=x2))

round(rms_preds,5) == round(trunc_power_preds,5)

rms_preds

trunc_power_preds

