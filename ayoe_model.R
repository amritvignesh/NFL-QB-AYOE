library(nflfastR)
library(dplyr)
library(xgboost)
library(tidyr)
library(caret)
library(vip)
library(DiagrammeR)
library(nflplotR)
library(ggrepel)
library(readr)
library(patchwork)

pbp <- load_pbp(2014:2023)

data <- pbp %>%
  filter(!is.na(air_yards)) %>%
  filter(play_type != "qb_spike") %>%
  mutate(timeouts = ifelse(posteam_type == "home", home_timeouts_remaining, away_timeouts_remaining)) %>%
  mutate(season_final = as.numeric(substring(game_id, 1, 4))) %>%
  filter(!is.na(receiver))

target_stats <- calculate_player_stats(pbp, weekly = TRUE)

target_stats <- subset(target_stats, player_id %in% data$receiver_id)

target_stats <- target_stats %>%
  select(player_id, player_display_name, season, week, receiving_epa)
colnames(data)

target_stats <- target_stats %>%
  arrange(player_id, week) %>%
  group_by(player_id) %>%
  select(receiver_id = player_id, season, week, receiving_epa)

unique_receivers <- unique(data$receiver_id)

final_target_stats <- data.frame(matrix(nrow = length(unique_receivers), ncol = 4))
colnames(final_target_stats) <- c("receiver_id", "season", "week", "receiving_epa")

final_target_stats$receiver_id <- unique_receivers

max_week <- max(target_stats$week)

final_target_stats <- expand.grid(receiver_id = final_target_stats$receiver_id, season = 2014:2023, week = 1:max_week)

final_target_stats <- left_join(final_target_stats, target_stats, by = c("receiver_id", "season", "week")) 

final_target_stats <- final_target_stats %>%
  mutate(receiving_epa = ifelse(is.na(receiving_epa), 0, receiving_epa)) %>%
  arrange(receiver_id, season, week) %>%
  group_by(receiver_id, season) %>%
  mutate(cumulative_receiving_epa = cumsum(receiving_epa))

new_data <- inner_join(data, final_target_stats, by = c("receiver_id", "season_final"="season", "week")) 

new_data <- new_data %>%
  select(passer_id, season_final, name, posteam, yardline_100, half_seconds_remaining, down, ydstogo, shotgun, no_huddle, qb_dropback, ep, wp, score_differential, cumulative_receiving_epa, air_yards)

colnames(new_data)
factor_data <- new_data %>%
  select(-passer_id, -season_final, -name, -posteam)

factor_data$down <- as.factor(factor_data$down)
factor_data$shotgun <- as.factor(factor_data$shotgun)
factor_data$no_huddle <- as.factor(factor_data$no_huddle)
factor_data$qb_dropback <- as.factor(factor_data$qb_dropback)

dummy <- dummyVars(" ~ .", data = factor_data)
final_data <- data.frame(predict(dummy, newdata = factor_data))

final_data <- cbind(new_data, final_data) 

final_data <- final_data[,-c(7, 9:11, 17:18, 23, 30:34)]

xgboost_train <- final_data %>%
  filter(season_final < 2021)

str(xgboost_train)

xgboost_test <- final_data %>%
  filter(season_final >= 2021)

labels_train <- as.matrix(xgboost_train[, 12])
xgboost_trainfinal <- as.matrix(xgboost_train[, c(5:11, 13:22)])
xgboost_testfinal <- as.matrix(xgboost_test[, c(5:11, 13:22)])

ayoe_model <- xgboost(data = xgboost_trainfinal, label = labels_train, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

vip(ayoe_model)

xgb.plot.tree(model = ayoe_model, trees = 1)

air_yards_predict <- predict(ayoe_model, xgboost_testfinal)
air_yards <- as.matrix(xgboost_test[,12])
postResample(air_yards_predict, air_yards)

air_yards_predictions <- as.data.frame(
  matrix(predict(ayoe_model, as.matrix(final_data[,c(5:11, 13:22)])))
)

all_stats <- cbind(final_data, air_yards_predictions) %>%
  select(passer_id, season = season_final, name, team = posteam, air_yards, pred_air_yards = V1)

all_stats <- all_stats %>%
  group_by(passer_id, season, name) %>%
  summarize(air_yards = sum(air_yards), pred_air_yards = sum(pred_air_yards), ayoe = air_yards - pred_air_yards)

attempts <- data %>%
  mutate(pass_attempt = ifelse(play_type == "pass", 1, 0)) %>%
  group_by(passer_id, season_final) %>%
  summarize(attempts = n())

all_stats <- inner_join(all_stats, attempts, by = c("passer_id", "season" = "season_final"))

all_stats <- all_stats %>%
  mutate(avg_air_yards = air_yards/attempts, avg_pred_air_yards = pred_air_yards/attempts, avg_ayoe = ayoe/attempts)

stats_2023 <- all_stats %>%
  filter(season == 2023, attempts >= 100) 

write_csv(stats_2023, "stats_2023.csv")

stats_2023 <- read_csv("stats_2023.csv")

stats_2023 %>%
  ggplot(aes(x = avg_air_yards, y = avg_pred_air_yards)) +
  geom_hline(yintercept = mean(stats_2023$avg_pred_air_yards), color = "red", linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = mean(stats_2023$avg_air_yards), color = "red", linetype = "dashed", alpha = 0.5) +
  geom_nfl_headshots(aes(player_gsis = passer_id), height = 0.07) +
  geom_text_repel(aes(label=name), size = 2.5, box.padding = 0.5, point.padding = 1.5) +
  geom_smooth(method = "lm") + 
  labs(x = "Average Air Yards",
       y = "Average Predicted Air Yards",
       title = "Predicting Air Yards and Quantifying AYOE",
       caption = "Amrit Vignesh") + 
  theme_bw() +
  theme(plot.title = element_text(size = 14, hjust = 0.5, face = "bold")) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 20)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 20))

gtdata <- stats_2023 %>%
  select(name, avg_air_yards, avg_pred_air_yards, ayoe) 
