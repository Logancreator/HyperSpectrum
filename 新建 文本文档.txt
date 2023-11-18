# randomForest
library(randomForest)
library(tidyverse)

data(airquality)
airquality <- airquality %>% na.omit()
airquality

# 数据训练
# 这里首先把数据集划分训练集（70%）和测试集（30%）


# To evaluate the performance of RF
# split traning data (70%) and validation data (30%)
set.seed(123)
train <- sample(nrow(airquality), nrow(airquality)*0.7)
ozo_train <- airquality[train, ]
ozo_test <- airquality[-train, ]
 
# 使用randomForest开始训练，其中Ozone~代表臭氧Ozone为因变量，其它数据为自变量。
# Random forest calculation（default 500 tress），please see ?randomForest
set.seed(123)
ozo_train.forest <- randomForest(Ozone~., data = ozo_train, importance = TRUE)
ozo_train.forest

# Scatterplot
library(ggplot2)
library(ggExtra)
library(ggpmisc)
library(ggpubr)

g <- ggplot(train_test, aes(obs, pre)) + 
  geom_point() + 
  geom_smooth(method="lm", se=F) +
  geom_abline(slope = 1,intercept = 0,lty="dashed") +
  stat_poly_eq(
    aes(label =paste( ..adj.rr.label.., sep = '~~')),
    formula = y ~ x,  parse = TRUE,
      family="serif",
      size = 6.4,
      color="black",
      label.x = 0.1,  #0-1之间的比例确定位置
      label.y = 1)

g1 <- ggMarginal(g, type = "histogram", fill="transparent")
g <- ggplot(predict_test, aes(obs, pre)) + 
  geom_point() + 
  geom_smooth(method="lm", se=F) +
  geom_abline(slope = 1,intercept = 0,lty="dashed") +
  stat_poly_eq(
    aes(label =paste( ..adj.rr.label.., sep = '~~')),
    formula = y ~ x,  parse = TRUE,
      family="serif",
      size = 6.4,
      color="black",
      label.x = 0.1,  #0-1之间的比例确定位置
      label.y = 1)

g2 <- ggMarginal(g, type = "histogram", fill="transparent")
ggarrange(g1, g2, ncol = 2)
# ggMarginal(g, type = "boxplot", fill="transparent")
# ggMarginal(g, type = "density", fill="transparent")


# 重要性评估
# 接下来查看变量重要性
##Ozo 的重要性评估
importance_ozo <- ozo_train.forest$importance
importance_ozo
importance_plot <- tibble(var = rownames(importance_ozo), 
                          IncMSE = importance_ozo[,1],
                          IncNodePurity = importance_ozo[,2])

# 对重要性排序进行可视化：

p1 <- ggplot(importance_plot, aes(x=var, y=IncMSE)) +
  geom_segment( aes(x=var, xend=var, y=0, yend=IncMSE), color="skyblue") +
  geom_point( color="blue", size=4, alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )
  p2 <- ggplot(importance_plot, aes(x=var, y=IncNodePurity)) +
  geom_segment( aes(x=var, xend=var, y=0, yend=IncNodePurity), color="skyblue") +
  geom_point( color="blue", size=4, alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )
ggarrange(p1, p2, ncol = 2)

# 接下来进行五折交叉验证，来选取超参数（这里是变量个数）：

# replicate用于重复n次所需语句，这里进行5次五折交叉验证
# rfcv通过嵌套交叉验证程序显示模型的交叉验证预测性能，模型的预测器数量按顺序减少（按变量重要性排序）。
# step如果log=TRUE，则为每个步骤要删除的变量的分数，否则一次删除这么多变量
# cv.fold为折数
#5 次重复五折交叉验证
set.seed(111)
ozo_train.cv <- replicate(5, rfcv(ozo_train[-ncol(ozo_train)], ozo_train$Ozone, cv.fold = 5, step = 0.8), simplify = FALSE)
#ozo_train.cv
ozo_train.cv <- data.frame(sapply(ozo_train.cv, '[[', 'error.cv'))
ozo_train.cv$vars <- rownames(ozo_train.cv)
ozo_train.cv <- reshape2::melt(ozo_train.cv, id = 'vars')
ozo_train.cv$vars <- as.numeric(as.character(ozo_train.cv$vars))
 
ozo_train.cv.mean <- aggregate(ozo_train.cv$value, by = list(ozo_train.cv$vars), FUN = mean)
ozo_train.cv.mean

# 可视化误差结果

ggplot(ozo_train.cv.mean, aes(Group.1, x)) +
    geom_line() +
    labs(title = '',x = 'Number of vars', y = 'Cross-validation error')


# 根据交叉验证曲线，提示保留1个重要的变量（或前四个重要的变量）获得理想的回归结果，因为此时的误差达到最小。

# 因此，根据计算得到的各ozone重要性的值（如“IncNodePurity”），将重要性由高往低排序后，最后大约选择前4个变量就可以了。

#首先根据某种重要性的高低排个序，例如根据“IncNodePurity”指标
importance_ozo <- importance_plot[order(importance_plot$IncNodePurity, decreasing = TRUE), ]
 
#然后取出排名靠前的因素
importance_ozo.select <- importance_ozo[1:4, ]
vars <- c(pull(importance_ozo.select, var), 'Ozone')
ozo.select <- airquality[ ,vars]
ozo.select <- reshape2::melt(ozo.select, id = 'Ozone')

# 查看下这些重要的 vars 与Ozone的关系
ggplot(ozo.select, aes(x = Ozone, y = value)) +
    geom_point() +
    geom_smooth() +
    facet_wrap(~variable, ncol = 2, scale = 'free_y') +
    labs(title = '',x = 'Ozone', y = 'Relative abundance')