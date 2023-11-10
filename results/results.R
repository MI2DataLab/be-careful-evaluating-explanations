library(xtable)
library(tidyr)
library(dplyr)
library(ggplot2)
# install.packages("ggsci")
library(ggsci)


## performance auc

df_auc <- read.csv("chexlocalize_aucroc.csv")
df_auc_long <- gather(df_auc, "model", "auc", DenseNet:Swin.ViT)
colnames(df_auc_long) <- c("class", "explanation", "pretrained", "scenario", "model", "auc")

df_auc_long %>% 
  group_by(pretrained, model, explanation, scenario) %>% 
  summarise(auc = mean(auc)) -> df_auc_aggregated

df_final <- pivot_wider(df_auc_aggregated, names_from = scenario, values_from = auc)
 
print(xtable(as.data.frame(df_final), type = "latex", digits = 3), 
      include.rownames = FALSE,
      booktabs = TRUE)

## alignment accuracy

df_acc <- read.csv("metrics_diff_results.csv")      

df_acc$xai_method <- factor(ifelse(df_acc$xai_method == "lrp", "LRP",
                                   ifelse(df_acc$xai_method == "integrated_gradients", "IG",
                                          ifelse(df_acc$xai_method == "gradient", "VG", "SG"))),
                            levels=c("VG", "IG", "SG", "LRP"))
df_acc$model <- factor(ifelse(df_acc$model == "vit", "ViT",
                              ifelse(df_acc$model == "swinvit", "Swin-ViT", "DenseNet")),
                       levels=c("DenseNet", "ViT", "Swin-ViT"))


### robustness

summary(lm(mass_accuracy_diff~model+xai_method+pretraining, data = df_acc))
summary(lm(rank_accuracy_diff~model+xai_method+pretraining, data = df_acc))


df_acc %>% group_by(model, xai_method, pretraining) %>%
  summarise(mass_accuracy_diff = mean(mass_accuracy_diff),
            rank_accuracy_diff = mean(rank_accuracy_diff)) -> df_acc_aggregated

print(as.data.frame(df_acc_aggregated[order(df_acc_aggregated$pretraining),]),
      digits=3)


### correlation table

cor(df_acc$in_mask_mass_accuracy, df_acc$in_mask_rank_accuracy, method = "pearson")
cor(df_acc$in_mask_mass_accuracy, df_acc$in_mask_rank_accuracy, method = "spearman")

cor(df_acc$out_mask_mass_accuracy, df_acc$out_mask_rank_accuracy, method = "pearson")
cor(df_acc$out_mask_mass_accuracy, df_acc$out_mask_rank_accuracy, method = "spearman")

cor(df_acc$in_mask_mass_accuracy, df_acc$out_mask_mass_accuracy, method = "pearson")
cor(df_acc$in_mask_mass_accuracy, df_acc$out_mask_mass_accuracy, method = "spearman")

cor(df_acc$in_mask_rank_accuracy, df_acc$out_mask_rank_accuracy, method = "pearson")
cor(df_acc$in_mask_rank_accuracy, df_acc$out_mask_rank_accuracy, method = "spearman")

cor(df_acc$in_mask_rank_accuracy, df_acc$out_mask_mass_accuracy, method = "pearson")
cor(df_acc$in_mask_rank_accuracy, df_acc$out_mask_mass_accuracy, method = "spearman")

cor(df_acc$in_mask_mass_accuracy, df_acc$out_mask_rank_accuracy, method = "pearson")
cor(df_acc$in_mask_mass_accuracy, df_acc$out_mask_rank_accuracy, method = "spearman")


### class labels aggregated + detailed

df_acc$label[df_acc$label == "Enlarged Cardiomediastinum"] <- "Enl. Card."

ggplot(df_acc %>% filter(pretraining == "pretrained", 
                         model == "ViT")) + 
  geom_point(aes(rank_accuracy_diff, mass_accuracy_diff,
                 color = label),
             size = 1) +
  facet_wrap(~xai_method, ncol = 4) +
  ggtitle("ViT combined with:") +
  theme_bw() + 
  theme(legend.position = "none") +
  scale_x_continuous(limits=c(-0.15, 0.45), expand=c(0,0), name = "Robustness: rank accuracy") +
  scale_y_continuous(limits=c(-0.15, 0.45), expand=c(0,0), name = "Robustness: mass accuracy") +
  scale_color_npg(alpha = 0.8)

ggsave("figures/labels_vit_all.pdf", width = 7, height = 2.5)

ggplot(df_acc %>% filter(pretraining == "pretrained", 
                           xai_method == "IG",
                           model == "ViT")) + 
    geom_point(aes(rank_accuracy_diff, mass_accuracy_diff, color = label),
               size = 1) +
    labs(#title="Details: ViT combined with IG",
         color = "Class label") + 
  theme_bw() + 
    theme(legend.position = "left") +
  scale_x_continuous(limits=c(-0.1, 0.31), expand=c(0,0),
                     name = NULL) + # name = "Robustness: rank accuracy") +
  scale_y_continuous(limits=c(-0.1, 0.31), expand=c(0,0), 
                     name = NULL) + # name = "Robustness: mass accuracy") + 
  scale_color_npg(alpha = 0.8)

ggsave("figures/labels_vit_ig.pdf", width = 5, height = 2.5)


### density of robustness

df_acc %>% 
  # filter(pretraining == "pretrained") %>%
  filter(pretraining != "pretrained") %>%
  mutate(model_explanation = factor(
    paste0(model, "–", xai_method),
    levels = apply(tidyr::expand_grid(levels(model), levels(xai_method)), 1, paste, collapse="–"))) %>%
  select(in_mask_rank_accuracy, out_mask_rank_accuracy, model_explanation) %>% gather(
  "scenario", "accuracy", in_mask_rank_accuracy:out_mask_rank_accuracy
  ) -> df_acc_aggregated_rank

df_acc_aggregated_rank$scenario <- factor(ifelse(df_acc_aggregated_rank$scenario == "in_mask_rank_accuracy", "Align", "Misalign"))

ggplot(df_acc_aggregated_rank) +
  geom_density(aes(accuracy, fill = scenario), 
               alpha = 0.5, bounds = c(0, 0.95)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 0.61)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_fill_manual(values = c("#4285F4", "#DB4437"), name = NULL) +
  facet_wrap(~model_explanation) +
  theme_bw() +
  theme(
    legend.position = c(0.93, 0.13),
        legend.title = element_text(size=1),
        text = element_text(size = 12),
        legend.text = element_text(size = 9),
        legend.background = element_rect(color = "black", 
                                         linewidth=0.5, linetype="solid"),
        panel.spacing.x = unit(0.55, "cm")) +
  labs(x = "Rank accuracy", y = "Density") -> p_rank

p_rank

# ggsave("figures/pipeline_accuracy_rank.pdf", p_rank, width = 7, height = 3.5)
ggsave("figures/pipeline_accuracy_rank_random.pdf", p_rank, width = 7, height = 3.5)


df_acc %>% 
  # filter(pretraining == "pretrained") %>%
  filter(pretraining != "pretrained") %>%
  mutate(model_explanation = factor(
    paste0(model, "–", xai_method),
    levels = apply(tidyr::expand_grid(levels(model), levels(xai_method)), 1, paste, collapse="–"))) %>%
  select(in_mask_mass_accuracy, out_mask_mass_accuracy, model_explanation) %>% gather(
    "scenario", "accuracy", in_mask_mass_accuracy:out_mask_mass_accuracy
  ) -> df_acc_aggregated_mass

df_acc_aggregated_mass$scenario <- factor(ifelse(df_acc_aggregated_mass$scenario == "in_mask_mass_accuracy", "Align", "Misalign"))

ggplot(df_acc_aggregated_mass) +
  geom_density(aes(accuracy, after_stat(count) / 12, fill = scenario), 
               alpha = 0.5, bounds = c(0, 0.9)) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 0.61)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 320)) +
  scale_fill_manual(values = c("#4285F4", "#DB4437")) +
  facet_wrap(~model_explanation) +
  theme_bw() +
  theme(
    legend.position = c(0.93, 0.13),
    legend.title = element_text(size=1),
    text = element_text(size = 12),
    legend.text = element_text(size = 9),
    legend.background = element_rect(color = "black", 
                                     size=0.5, linetype="solid"),
    panel.spacing.x = unit(0.55, "cm")) +
  labs(x = "Mass accuracy", y = "Density", fill = NULL) -> p_mass

p_mass

# ggsave("figures/pipeline_accuracy_mass.pdf", p_mass, width = 7, height = 3.5)
ggsave("figures/pipeline_accuracy_mass_random.pdf", p_mass, width = 7, height = 3.5)

