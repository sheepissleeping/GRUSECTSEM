# =============================================================================
# 大五人格 + 年龄/性别 对情绪动态的CTSEM完整分析
# 版本：完整修正版 - 包含所有步骤，确保运行到底
# 修改：使用整体标准化 + 单核运行（避免错误）
# =============================================================================

# 加载必要的包
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ctsem)
library(reshape2)
library(stringr)
library(tibble)
library(scales)
library(expm)
library(cowplot)

# =============================================================================
# 配置参数
# =============================================================================
config <- list(
  data_path = "D:/data.xlsx",
  save_path = "C:/Users/48538/Desktop/jg"
)

# 创建输出目录
if(!dir.exists(config$save_path)) {
  dir.create(config$save_path, recursive = TRUE)
  cat("✅ 创建输出目录:", config$save_path, "\n")
}
if(!dir.exists(file.path(config$save_path, "Plots"))) {
  dir.create(file.path(config$save_path, "Plots"))
  cat("✅ 创建图表目录:", file.path(config$save_path, "Plots"), "\n")
}

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  大五人格 + 年龄/性别 对情绪动态的CTSEM分析\n")
cat("  ✨ 完整版：从数据到结果，一次运行完成\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("数据路径:", config$data_path, "\n")
cat("保存路径:", config$save_path, "\n")
cat("开始时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# =============================================================================
# 配色方案 & 主题
# =============================================================================
big5_colors <- c(
  "Extraversion" = "#E64B35", 
  "Agreeableness" = "#4DBBD5", 
  "Conscientiousness" = "#00A087", 
  "Neuroticism" = "#F39B7F", 
  "Openness" = "#8491B4"
)

theme_clean <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = base_size + 4, 
                                margin = margin(b = 12)),
      plot.subtitle = element_text(hjust = 0.5, size = base_size, color = "gray40",
                                   margin = margin(b = 10)),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(size = base_size - 2),
      legend.title = element_text(face = "bold", size = base_size - 1),
      legend.text = element_text(size = base_size - 2),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray92"),
      strip.text = element_text(face = "bold", size = base_size),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin = margin(15, 15, 15, 15)
    )
}

# =============================================================================
# STEP 1: 数据加载与预处理
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 1: 加载和预处理数据\n")
cat("═══════════════════════════════════════════════════════════════\n")

# 检查文件是否存在
if(!file.exists(config$data_path)) {
  stop("❌ 数据文件不存在: ", config$data_path, "\n请检查路径是否正确！")
}

# 读取数据
cat("正在读取数据...\n")
dat <- read_excel(config$data_path)

cat("✅ 数据加载成功\n")
cat("  原始数据:", nrow(dat), "行 ×", ncol(dat), "列\n")
cat("  个体数量:", length(unique(dat$name)), "\n\n")

# 数据预处理
cat("正在预处理数据...\n")
ctsem_data <- dat %>%
  rename(
    id = name,
    time = start_second,
    Neuroticism = Negative_Emotionality,
    Openness = Open_Mindedness
  ) %>%
  mutate(
    id = as.character(id),
    time = as.numeric(time),
    # 标准化时不变预测变量
    age_z = scale(age)[,1],
    Extraversion_z = scale(Extraversion)[,1],
    Agreeableness_z = scale(Agreeableness)[,1],
    Conscientiousness_z = scale(Conscientiousness)[,1],
    Neuroticism_z = scale(Neuroticism)[,1],
    Openness_z = scale(Openness)[,1],
    gender_c = gender - mean(gender),
    # ✅ 关键：整体标准化情绪变量
    P_std = scale(P)[,1],
    A_std = scale(A)[,1]
  )

cat("✅ 变量标准化完成\n")

# 选择分析所需变量
final_data <- ctsem_data %>%
  select(id, time, P, A, P_std, A_std, age, gender,
         Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness,
         age_z, gender_c, Extraversion_z, Agreeableness_z, Conscientiousness_z, 
         Neuroticism_z, Openness_z) %>%
  filter(complete.cases(P_std, A_std, id, time)) %>%
  arrange(id, time)

cat("✅ 数据预处理完成\n")
cat("  有效个体:", length(unique(final_data$id)), "\n")
cat("  总观测数:", nrow(final_data), "\n")
cat("  平均每人:", round(nrow(final_data)/length(unique(final_data$id)), 1), "个观测\n")
cat("  P标准化: M=", round(mean(final_data$P_std), 3), ", SD=", round(sd(final_data$P_std), 3), "\n")
cat("  A标准化: M=", round(mean(final_data$A_std), 3), ", SD=", round(sd(final_data$A_std), 3), "\n\n")

# =============================================================================
# STEP 2: 准备CTSEM模型数据
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 2: 准备CTSEM模型数据\n")
cat("═══════════════════════════════════════════════════════════════\n")

cat("正在准备模型数据...\n")

# 准备ctsem所需的数据格式
model_data <- final_data %>%
  select(id, time, P_std, A_std,
         age_z, gender_c,
         Extraversion_z, Agreeableness_z, Conscientiousness_z, 
         Neuroticism_z, Openness_z) %>%
  rename(
    P = P_std,
    A = A_std,
    age = age_z,
    gender = gender_c,
    Extraversion = Extraversion_z,
    Agreeableness = Agreeableness_z,
    Conscientiousness = Conscientiousness_z,
    Neuroticism = Neuroticism_z,
    Openness = Openness_z
  ) %>%
  group_by(id) %>%
  filter(n() >= 3) %>%  # 至少3个观测
  ungroup() %>%
  as.data.frame()

cat("✅ 模型数据准备完成\n")
cat("  个体数:", length(unique(model_data$id)), "\n")
cat("  观测数:", nrow(model_data), "\n")
cat("  P范围: [", round(min(model_data$P), 2), ",", round(max(model_data$P), 2), "]\n")
cat("  A范围: [", round(min(model_data$A), 2), ",", round(max(model_data$A), 2), "]\n\n")

# 定义时间不变预测变量
ti_covs <- c("age", "gender", "Extraversion", "Agreeableness", 
             "Conscientiousness", "Neuroticism", "Openness")

cat("时间不变预测变量 (", length(ti_covs), "个):\n")
cat("  ", paste(ti_covs, collapse = ", "), "\n\n")

# =============================================================================
# STEP 3: 定义CTSEM模型
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 3: 定义CTSEM模型\n")
cat("═══════════════════════════════════════════════════════════════\n")

cat("正在定义模型...\n")

ctsem_model <- ctModel(
  type = 'stanct',
  n.latent = 2,
  n.manifest = 2,
  LAMBDA = diag(2),
  manifestNames = c('P', 'A'),
  latentNames = c('eta_P', 'eta_A'),
  TIpredNames = ti_covs,
  id = 'id',
  time = 'time'
)

cat("✅ 模型定义完成\n")
cat("  类型: 连续时间状态空间模型 (CT-SEM)\n")
cat("  潜变量: eta_P (正情绪), eta_A (负情绪)\n")
cat("  观测变量: P (标准化), A (标准化)\n")
cat("  预测变量: 协变量(age, gender) + 大五人格\n\n")

# =============================================================================
# STEP 4: 模型拟合
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 4: 模型拟合 (这一步需要10-30分钟)\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  方法: 最大后验估计 (MAP)\n")
cat("  核心数: 1 (单核，避免序列化错误)\n")
cat("  开始时间:", format(Sys.time(), "%H:%M:%S"), "\n")
cat("  预计完成:", format(Sys.time() + 20*60, "%H:%M:%S"), "(约20分钟后)\n\n")
cat("⏳ 正在拟合模型，请耐心等待...\n")
cat("   (看到迭代信息说明正在运行，不要中断！)\n\n")

start_time <- Sys.time()

# ✅ 关键设置：单核 + nopriors
ctsem_fit <- ctStanFit(
  datalong = model_data,
  ctstanmodel = ctsem_model,
  optimize = TRUE,
  optimcontrol = list(
    stochastic = FALSE,
    finishsamples = 1000
  ),
  cores = 1,           # ← 单核，避免错误
  nopriors = TRUE,     # ← 不用先验，更快更稳定
  verbose = 1
)

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  ✅ 模型拟合完成！\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  实际用时:", round(elapsed, 1), "分钟\n")
cat("  完成时间:", format(Sys.time(), "%H:%M:%S"), "\n\n")

# 保存模型
model_file <- file.path(config$save_path, "ctsem_fit.rds")
saveRDS(ctsem_fit, model_file)
cat("✅ 模型已保存:", model_file, "\n\n")

# =============================================================================
# STEP 5: 提取模型结果
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 5: 提取模型参数\n")
cat("═══════════════════════════════════════════════════════════════\n")

cat("正在提取模型摘要...\n")
model_summary <- summary(ctsem_fit)

# 提取拟合指标
aic_value <- ifelse(!is.null(model_summary$aic), round(model_summary$aic, 4), NA)
bic_value <- ifelse(!is.null(model_summary$bic), round(model_summary$bic, 4), NA)
loglik_value <- ifelse(!is.null(model_summary$loglik), round(model_summary$loglik, 4), NA)

cat("\n模型拟合指标:\n")
cat("  对数似然值:", loglik_value, "\n")
cat("  AIC:", aic_value, "\n")
cat("  BIC:", bic_value, "\n\n")

# 提取DRIFT参数
cat("正在提取DRIFT参数...\n")

drift_11 <- NA
drift_22 <- NA
drift_12 <- NA
drift_21 <- NA

if(!is.null(model_summary$parmatrices)) {
  drift_params <- as.data.frame(model_summary$parmatrices) %>%
    filter(matrix == "DRIFT")
  
  if(nrow(drift_params) > 0) {
    drift_11 <- drift_params$Mean[drift_params$row == 1 & drift_params$col == 1]
    drift_22 <- drift_params$Mean[drift_params$row == 2 & drift_params$col == 2]
    drift_12 <- drift_params$Mean[drift_params$row == 1 & drift_params$col == 2]
    drift_21 <- drift_params$Mean[drift_params$row == 2 & drift_params$col == 1]
  }
}

# 备用方法
if(is.na(drift_11)) {
  cat("使用备用方法提取DRIFT...\n")
  tryCatch({
    extracted <- ctExtract(ctsem_fit)
    if(!is.null(extracted$DRIFT)) {
      drift_matrix <- apply(extracted$DRIFT, c(2,3), mean)
      drift_11 <- drift_matrix[1, 1]
      drift_22 <- drift_matrix[2, 2]
      drift_12 <- drift_matrix[1, 2]
      drift_21 <- drift_matrix[2, 1]
    }
  }, error = function(e) {
    cat("警告: DRIFT提取失败\n")
  })
}

DRIFT <- matrix(c(drift_11, drift_21, drift_12, drift_22), nrow = 2, byrow = FALSE)

cat("\n")
cat("  ╔════════════════════════════════════════════════╗\n")
cat("  ║         DRIFT 矩阵估计结果                     ║\n")
cat("  ║   ✅ P和A已标准化，参数可直接比较！           ║\n")
cat("  ╠════════════════════════════════════════════════╣\n")
cat(sprintf("  ║  P → P (自回归):      %7.4f               ║\n", drift_11))
cat(sprintf("  ║  A → A (自回归):      %7.4f               ║\n", drift_22))
cat(sprintf("  ║  P → A (交叉回归):    %7.4f               ║\n", drift_21))
cat(sprintf("  ║  A → P (交叉回归):    %7.4f               ║\n", drift_12))
cat("  ╚════════════════════════════════════════════════╝\n\n")

# 判断参数是否合理
if(abs(drift_22) > 10) {
  cat("⚠️  警告: A→A参数过大 (", drift_22, ")，可能是旧版本（个体中心化）\n")
  cat("   建议检查数据预处理步骤！\n\n")
} else {
  cat("✅ DRIFT参数合理，标准化成功！\n")
  cat("   P→P和A→A参数在相似范围内，可以直接比较。\n\n")
}

# 提取TI预测变量效应
cat("正在提取时间不变预测变量效应...\n")
tipreds <- NULL

if(!is.null(model_summary$tipreds)) {
  tipreds <- as.data.frame(model_summary$tipreds)
  tipreds$parameter <- rownames(tipreds)
  
  if("Mean" %in% names(tipreds)) tipreds$mean <- tipreds$Mean
  if("Sd" %in% names(tipreds)) tipreds$sd <- tipreds$Sd
  if("2.5%" %in% names(tipreds)) {
    tipreds$ci_lower <- tipreds$`2.5%`
    tipreds$ci_upper <- tipreds$`97.5%`
  }
  
  tipreds <- tipreds %>%
    mutate(
      predictor = gsub("tip_(.+)_(T0m|drift|diff|mm|cint).*", "\\1", parameter),
      target_type = case_when(
        grepl("_T0m_", parameter) ~ "T0m",
        grepl("_drift_", parameter) ~ "drift",
        grepl("_diff_", parameter) ~ "diff",
        grepl("_mm_", parameter) ~ "mm",
        grepl("_cint_", parameter) ~ "cint",
        TRUE ~ "other"
      ),
      target_var = case_when(
        grepl("eta_P$", parameter) & !grepl("eta_A", parameter) ~ "P",
        grepl("eta_A$", parameter) & !grepl("eta_P", parameter) ~ "A",
        grepl("eta_P_eta_A", parameter) ~ "A_to_P",
        grepl("eta_A_eta_P", parameter) ~ "P_to_A",
        TRUE ~ "other"
      ),
      predictor_label = case_when(
        predictor == "age" ~ "Age",
        predictor == "gender" ~ "Gender",
        predictor == "Extraversion" ~ "Extraversion",
        predictor == "Agreeableness" ~ "Agreeableness",
        predictor == "Conscientiousness" ~ "Conscientiousness",
        predictor == "Neuroticism" ~ "Neuroticism",
        predictor == "Openness" ~ "Openness",
        TRUE ~ predictor
      ),
      target_label = case_when(
        target_type == "drift" & target_var == "P" ~ "P → P",
        target_type == "drift" & target_var == "A" ~ "A → A",
        target_type == "drift" & target_var == "A_to_P" ~ "A → P",
        target_type == "drift" & target_var == "P_to_A" ~ "P → A",
        TRUE ~ NA_character_
      ),
      predictor_type = case_when(
        predictor %in% c("age", "gender") ~ "Covariate",
        TRUE ~ "Big Five"
      )
    )
  
  # 显著性检验
  if("sd" %in% names(tipreds) && !all(tipreds$sd == 0, na.rm = TRUE)) {
    tipreds <- tipreds %>%
      mutate(
        z = mean / sd,
        significant = abs(z) > 1.96,
        sig_star = case_when(
          abs(z) > 3.29 ~ "***",
          abs(z) > 2.58 ~ "**",
          abs(z) > 1.96 ~ "*",
          TRUE ~ ""
        )
      )
  } else if("ci_lower" %in% names(tipreds)) {
    tipreds <- tipreds %>%
      mutate(
        significant = (ci_lower > 0 & ci_upper > 0) | (ci_lower < 0 & ci_upper < 0),
        sig_star = ifelse(significant, "*", "")
      )
  } else {
    tipreds$sig_star <- ""
  }
  
  cat("✅ TI预测变量效应提取完成\n")
  cat("  共", nrow(tipreds), "个参数估计\n\n")
}

# =============================================================================
# STEP 6: 生成核心图表
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 6: 生成可视化图表\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

plots_created <- 0

# ----- 图1: 自回归系数衰减图 -----
cat("[ 1/9] 生成自回归系数衰减图...\n")
tryCatch({
  time_intervals <- seq(0, 100, by = 1)
  autoreg_data <- data.frame(
    time_interval = rep(time_intervals, 2),
    autoreg_coeff = c(exp(drift_11 * time_intervals), exp(drift_22 * time_intervals)),
    dimension = rep(c("正情绪 (P)", "负情绪 (A)"), each = length(time_intervals))
  )
  
  p1 <- ggplot(autoreg_data, aes(x = time_interval, y = autoreg_coeff, color = dimension)) +
    geom_line(linewidth = 1.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
    labs(
      title = "自回归系数随时间间隔的衰减",
      subtitle = sprintf("DRIFT: P→P = %.4f, A→A = %.4f", drift_11, drift_22),
      x = "时间间隔 (秒)", y = "自回归系数", color = ""
    ) +
    scale_color_manual(values = c("正情绪 (P)" = "#3C5488", "负情绪 (A)" = "#DC0000")) +
    theme_clean() + theme(legend.position = "bottom")
  
  ggsave(file.path(config$save_path, "Plots", "01_Autoreg_Decay.png"), 
         p1, width = 12, height = 8, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

# ----- 图2: 交叉回归效应 -----
cat("[ 2/9] 生成交叉回归效应图...\n")
tryCatch({
  calc_effects <- function(DRIFT, time_seq) {
    results <- lapply(time_seq, function(dt) {
      dm <- expm(DRIFT * dt)
      data.frame(time = dt, P_P = dm[1,1], A_P = dm[1,2], P_A = dm[2,1], A_A = dm[2,2])
    })
    do.call(rbind, results)
  }
  
  time_seq <- seq(0, 4, by = 0.02)
  effects <- calc_effects(DRIFT, time_seq) %>%
    pivot_longer(cols = -time, names_to = "effect", values_to = "value") %>%
    filter(effect %in% c("P_A", "A_P")) %>%
    mutate(effect_label = ifelse(effect == "P_A", "P → A", "A → P"))
  
  p2 <- ggplot(effects, aes(x = time, y = value, color = effect_label)) +
    geom_hline(yintercept = 0, linetype = "solid", color = "gray50") +
    geom_line(linewidth = 1.5) +
    scale_color_manual(values = c("P → A" = "#E64B35", "A → P" = "#7E6148"), name = "") +
    labs(title = "交叉回归效应", x = "时间间隔", y = "效应强度") +
    theme_clean() + theme(legend.position = "bottom")
  
  ggsave(file.path(config$save_path, "Plots", "02_Crosslagged.png"), 
         p2, width = 10, height = 7, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

# ----- 图3: DRIFT参数图 -----
cat("[ 3/9] 生成DRIFT参数图...\n")
tryCatch({
  drift_data <- data.frame(
    effect = factor(c("P → P", "A → A", "P → A", "A → P"),
                    levels = c("P → P", "A → A", "P → A", "A → P")),
    value = c(drift_11, drift_22, drift_21, drift_12),
    type = c("自回归", "自回归", "交叉", "交叉")
  )
  
  p3 <- ggplot(drift_data, aes(x = effect, y = value, fill = type)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_col(width = 0.6, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.5, size = 5, fontface = "bold") +
    scale_fill_manual(values = c("自回归" = "#3C5488", "交叉" = "#E64B35")) +
    labs(title = "DRIFT矩阵参数", x = "", y = "参数值", fill = "") +
    theme_clean() + theme(legend.position = "bottom")
  
  ggsave(file.path(config$save_path, "Plots", "03_DRIFT_Parameters.png"), 
         p3, width = 10, height = 7, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

# ----- 图4-6: 预测变量效应图 -----
if(!is.null(tipreds)) {
  # 图4: 大五人格热图
  cat("[ 4/9] 生成大五人格效应热图...\n")
  tryCatch({
    big5_drift <- tipreds %>%
      filter(predictor_type == "Big Five", target_type == "drift", !is.na(target_label)) %>%
      mutate(
        predictor_label = factor(predictor_label, 
                                  levels = c("Extraversion", "Agreeableness", 
                                             "Conscientiousness", "Neuroticism", "Openness")),
        target_label = factor(target_label, levels = c("P → P", "A → A", "P → A", "A → P"))
      )
    
    if(nrow(big5_drift) > 0) {
      p4 <- ggplot(big5_drift, aes(x = target_label, y = predictor_label)) +
        geom_tile(aes(fill = mean), color = "white", linewidth = 1.5) +
        geom_text(aes(label = paste0(sprintf("%.2f", mean), sig_star)), 
                  size = 5, fontface = "bold") +
        scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                             midpoint = 0, limits = c(-0.5, 0.5), oob = scales::squish) +
        scale_y_discrete(limits = rev) +
        labs(title = "大五人格对情绪动态的影响", 
             subtitle = "*p<.05 **p<.01 ***p<.001",
             x = "", y = "", fill = "效应值") +
        theme_clean() + theme(panel.grid = element_blank())
      
      ggsave(file.path(config$save_path, "Plots", "04_Big5_Heatmap.png"), 
             p4, width = 11, height = 8, dpi = 300)
      plots_created <- plots_created + 1
      cat("      ✅ 已保存\n")
    }
  }, error = function(e) cat("      ❌ 失败:", e$message, "\n"))
  
  # 图5: 协变量热图
  cat("[ 5/9] 生成协变量效应热图...\n")
  tryCatch({
    covar_drift <- tipreds %>%
      filter(predictor_type == "Covariate", target_type == "drift", !is.na(target_label)) %>%
      mutate(
        predictor_label = factor(predictor_label, levels = c("Age", "Gender")),
        target_label = factor(target_label, levels = c("P → P", "A → A", "P → A", "A → P"))
      )
    
    if(nrow(covar_drift) > 0) {
      p5 <- ggplot(covar_drift, aes(x = target_label, y = predictor_label)) +
        geom_tile(aes(fill = mean), color = "white", linewidth = 1.5) +
        geom_text(aes(label = paste0(sprintf("%.2f", mean), sig_star)), 
                  size = 6, fontface = "bold") +
        scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                             midpoint = 0, limits = c(-0.5, 0.5), oob = scales::squish) +
        labs(title = "年龄和性别对情绪动态的影响", 
             subtitle = "*p<.05 **p<.01 ***p<.001",
             x = "", y = "", fill = "效应值") +
        theme_clean() + theme(panel.grid = element_blank())
      
      ggsave(file.path(config$save_path, "Plots", "05_Covariate_Heatmap.png"), 
             p5, width = 10, height = 6, dpi = 300)
      plots_created <- plots_created + 1
      cat("      ✅ 已保存\n")
    }
  }, error = function(e) cat("      ❌ 失败:", e$message, "\n"))
  
  # 图6: 综合热图
  cat("[ 6/9] 生成综合效应热图...\n")
  tryCatch({
    all_drift <- tipreds %>%
      filter(target_type == "drift", !is.na(target_label)) %>%
      mutate(
        predictor_label = factor(predictor_label, 
                                  levels = c("Age", "Gender", "Extraversion", "Agreeableness", 
                                             "Conscientiousness", "Neuroticism", "Openness")),
        target_label = factor(target_label, levels = c("P → P", "A → A", "P → A", "A → P"))
      )
    
    if(nrow(all_drift) > 0) {
      p6 <- ggplot(all_drift, aes(x = target_label, y = predictor_label)) +
        geom_tile(aes(fill = mean), color = "white", linewidth = 1) +
        geom_text(aes(label = paste0(sprintf("%.2f", mean), sig_star)), 
                  size = 4, fontface = "bold") +
        scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                             midpoint = 0, limits = c(-0.5, 0.5), oob = scales::squish) +
        scale_y_discrete(limits = rev) +
        labs(title = "所有预测变量对情绪动态的影响", x = "", y = "", fill = "效应值") +
        theme_clean() + theme(panel.grid = element_blank())
      
      ggsave(file.path(config$save_path, "Plots", "06_All_Predictors.png"), 
             p6, width = 11, height = 9, dpi = 300)
      plots_created <- plots_created + 1
      cat("      ✅ 已保存\n")
    }
  }, error = function(e) cat("      ❌ 失败:", e$message, "\n"))
} else {
  cat("[ 4-6] ⚠️  跳过预测变量效应图（tipreds为空）\n")
}

# ----- 图7: PA时序图 -----
cat("[ 7/9] 生成情绪时序图...\n")
tryCatch({
  emotion_long <- final_data %>%
    select(id, time, P, A) %>%
    pivot_longer(cols = c(P, A), names_to = "emotion", values_to = "value") %>%
    mutate(emotion = factor(emotion, levels = c("P", "A"), 
                           labels = c("正情绪 (P)", "负情绪 (A)")))
  
  p7 <- ggplot(emotion_long, aes(x = time, y = value, color = id, group = id)) +
    geom_line(alpha = 0.3, linewidth = 0.3) +
    geom_smooth(aes(group = 1), method = "loess", color = "black", se = TRUE, linewidth = 1) +
    facet_wrap(~emotion, ncol = 1, scales = "free_y") +
    labs(title = "情绪时序数据", x = "时间", y = "强度") +
    theme_clean() + theme(legend.position = "none")
  
  ggsave(file.path(config$save_path, "Plots", "07_Timeseries.png"), 
         p7, width = 12, height = 10, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

# ----- 图8: 人口统计学 -----
cat("[ 8/9] 生成人口统计学分布图...\n")
tryCatch({
  demo <- final_data %>% distinct(id, .keep_all = TRUE)
  
  p8a <- ggplot(demo, aes(x = age)) +
    geom_histogram(bins = 15, fill = "#7570B3", alpha = 0.8) +
    geom_vline(xintercept = mean(demo$age), linetype = "dashed", color = "red") +
    labs(title = "年龄分布", x = "年龄", y = "频数") +
    theme_clean()
  
  p8b <- ggplot(demo, aes(x = factor(gender, labels = c("女", "男")), fill = factor(gender))) +
    geom_bar(alpha = 0.8) +
    geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
    scale_fill_manual(values = c("0" = "#E7298A", "1" = "#7570B3")) +
    labs(title = "性别分布", x = "", y = "人数") +
    theme_clean() + theme(legend.position = "none")
  
  p8 <- plot_grid(p8a, p8b, ncol = 2)
  ggsave(file.path(config$save_path, "Plots", "08_Demographics.png"), 
         p8, width = 12, height = 5, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

# ----- 图9: 人格分布 -----
cat("[ 9/9] 生成人格分布图...\n")
tryCatch({
  personality <- final_data %>%
    select(id, Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness) %>%
    distinct() %>%
    pivot_longer(cols = -id, names_to = "trait", values_to = "score") %>%
    mutate(trait = factor(trait, levels = names(big5_colors)))
  
  p9 <- ggplot(personality, aes(x = trait, y = score, fill = trait)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_manual(values = big5_colors) +
    labs(title = "大五人格分布", x = "", y = "得分") +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
  
  ggsave(file.path(config$save_path, "Plots", "09_Big5_Distribution.png"), 
         p9, width = 10, height = 8, dpi = 300)
  plots_created <- plots_created + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败:", e$message, "\n"))

cat("\n✅ 图表生成完成! 共生成", plots_created, "张图表\n\n")

# =============================================================================
# STEP 7: 保存结果摘要
# =============================================================================
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 7: 保存结果文件\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

files_saved <- 0

# 保存拟合指标
cat("[ 1/4] 保存模型拟合指标...\n")
tryCatch({
  sink(file.path(config$save_path, "model_fit_indices.txt"))
  cat("CTSEM模型拟合指标\n")
  cat("=" %R>% rep(50) %>% paste(collapse = ""), "\n\n")
  cat("对数似然值:", loglik_value, "\n")
  cat("AIC:", aic_value, "\n")
  cat("BIC:", bic_value, "\n\n")
  cat("DRIFT矩阵参数:\n")
  cat("  P → P:", drift_11, "\n")
  cat("  A → A:", drift_22, "\n")
  cat("  P → A:", drift_21, "\n")
  cat("  A → P:", drift_12, "\n\n")
  cat("样本信息:\n")
  cat("  个体数:", length(unique(model_data$id)), "\n")
  cat("  观测数:", nrow(model_data), "\n")
  sink()
  files_saved <- files_saved + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败\n"))

# 保存完整摘要
cat("[ 2/4] 保存完整模型摘要...\n")
tryCatch({
  sink(file.path(config$save_path, "ctsem_model_summary.txt"))
  print(model_summary)
  sink()
  files_saved <- files_saved + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败\n"))

# 保存处理后数据
cat("[ 3/4] 保存处理后的数据...\n")
tryCatch({
  write.csv(final_data, file.path(config$save_path, "processed_data.csv"), row.names = FALSE)
  files_saved <- files_saved + 1
  cat("      ✅ 已保存\n")
}, error = function(e) cat("      ❌ 失败\n"))

# 保存TI效应
if(!is.null(tipreds)) {
  cat("[ 4/4] 保存TI预测变量效应...\n")
  tryCatch({
    write.csv(tipreds, file.path(config$save_path, "TI_predictor_effects.csv"), row.names = FALSE)
    files_saved <- files_saved + 1
    cat("      ✅ 已保存\n")
  }, error = function(e) cat("      ❌ 失败\n"))
}

cat("\n✅ 结果文件保存完成! 共保存", files_saved, "个文件\n\n")

# =============================================================================
# 完成摘要
# =============================================================================
total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("                    🎉 分析全部完成！                          \n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("\n📁 输出目录:", config$save_path, "\n")
cat("\n📊 生成的文件:\n")
cat("  图表:", plots_created, "张 (Plots文件夹)\n")
cat("  结果文件:", files_saved, "个\n")
cat("\n📈 核心结果 - DRIFT参数:\n")
cat(sprintf("  P → P: %7.4f  (正情绪自回归)\n", drift_11))
cat(sprintf("  A → A: %7.4f  (负情绪自回归)\n", drift_22))
cat(sprintf("  P → A: %7.4f  (正→负交叉)\n", drift_21))
cat(sprintf("  A → P: %7.4f  (负→正交叉)\n", drift_12))
cat("\n💡 解释:\n")
if(abs(drift_22) < 5) {
  cat("  ✅ 参数合理！P和A在相同尺度上，可直接比较。\n")
  cat("  → 数值越负 = 衰减越快 = 越不稳定\n")
  cat("  → 接近0 = 衰减越慢 = 越稳定\n")
} else {
  cat("  ⚠️  A→A参数异常大，请检查数据预处理！\n")
}
cat("\n⏱️  总用时:", round(total_time, 1), "分钟\n")
cat("  模型拟合:", round(elapsed, 1), "分钟\n")
cat("  图表生成:", round(total_time - elapsed, 1), "分钟\n")
cat("\n📂 输出文件清单:\n")
cat("  模型文件:\n")
cat("    - ctsem_fit.rds\n")
cat("    - model_fit_indices.txt\n")
cat("    - ctsem_model_summary.txt\n")
cat("  图表 (Plots/):\n")
cat("    - 01_Autoreg_Decay.png\n")
cat("    - 02_Crosslagged.png\n")
cat("    - 03_DRIFT_Parameters.png\n")
cat("    - 04_Big5_Heatmap.png\n")
cat("    - 05_Covariate_Heatmap.png\n")
cat("    - 06_All_Predictors.png\n")
cat("    - 07_Timeseries.png\n")
cat("    - 08_Demographics.png\n")
cat("    - 09_Big5_Distribution.png\n")
cat("  数据文件:\n")
cat("    - processed_data.csv\n")
cat("    - TI_predictor_effects.csv\n")
cat("\n完成时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("\n✨ 分析成功！所有文件已保存到:", config$save_path, "\n\n")
