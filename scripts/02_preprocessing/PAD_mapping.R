# =============================================================================
# PAD映射脚本：将7类离散情绪映射为愉悦度(P)和唤醒度(A)
# 基于Russell环形情绪模型和PAD三维映射技术
# =============================================================================

# PAD映射表（基于表2）
PAD_mapping <- data.frame(
  Emotion_Type = c("Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"),
  Pleasure = c(-0.51, -0.4, -0.64, 0.4, -0.4, 0.2, 0),
  Arousal = c(0.59, 0.2, 0.6, 0.2, -0.2, 0.45, 0)
)

# 映射函数
map_emotion_to_PA <- function(emotion_labels) {
  # emotion_labels: 情绪标签向量（如 "ang", "hap", "neu"等）
  
  # 标签规范化映射
  label_map <- c(
    "ang" = "Anger",
    "anger" = "Anger",
    "dis" = "Disgust", 
    "disgust" = "Disgust",
    "fea" = "Fear",
    "fear" = "Fear",
    "hap" = "Happy",
    "happy" = "Happy",
    "sad" = "Sad",
    "sur" = "Surprise",
    "surprise" = "Surprise",
    "neu" = "Neutral",
    "neutral" = "Neutral"
  )
  
  # 转换为标准名称
  emotion_standard <- label_map[tolower(emotion_labels)]
  
  # 映射到P和A值
  result <- data.frame(
    emotion_label = emotion_labels,
    P = PAD_mapping$Pleasure[match(emotion_standard, PAD_mapping$Emotion_Type)],
    A = PAD_mapping$Arousal[match(emotion_standard, PAD_mapping$Emotion_Type)]
  )
  
  return(result)
}

# 使用示例
# 假设您有一列情绪标签
# emotion_data <- data.frame(label = c("ang", "hap", "neu", "sad"))
# PA_values <- map_emotion_to_PA(emotion_data$label)
# emotion_data <- cbind(emotion_data, PA_values[, c("P", "A")])

# 保存映射表
write.csv(PAD_mapping, "data/PAD_mapping.csv", row.names = FALSE)
cat("✅ PAD映射表已保存到 data/PAD_mapping.csv\n")
