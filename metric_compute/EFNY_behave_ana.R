rm(list = ls())
library(dplyr)
library(purrr)
library(readxl)
library(stringr)
library(tidyr)
library(tibble)

source("/Users/tanlirou/Documents/EFNY/data/code/behave_function.R") 

################define task################
task_config <- list(
  oneback = list(
    fun        = analyze_1back,
    rt_var     = "相对时间(秒)",
    rt_max     = 2.5,
    min_prop   = 0.5,
    ans_var    = "正式阶段正确答案",
    resp_var   = "正式阶段被试按键",
    filter_rt = TRUE,
    stim_var   = "正式阶段刺激图片/Item名" 
  ),
  twoback = list(
    fun        = analyze_2back,
    rt_var     = "相对时间(秒)",
    rt_max     = 2.5,
    min_prop   = 0.5,
    ans_var    = "正式阶段正确答案",
    resp_var   = "正式阶段被试按键",
    filter_rt = TRUE,
    stim_var   = "正式阶段刺激图片/Item名" 
  ),
  KT = list(
    fun      = analyze_keeptrack,
    rt_var   = "相对时间(秒)",
    rt_max   = 3.0,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    filter_rt = FALSE
  ),
  FLANKER = list(
    fun      = analyze_flanker_stroop,
    rt_var   = "相对时间(秒)",
    rt_max   = 2.0,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    item_var = "正式阶段刺激图片/Item名",
    filter_rt = TRUE
  ),
  ColorStroop = list(
    fun      = analyze_flanker_stroop,
    rt_var   = "相对时间(秒)",
    rt_max   = 2.5,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    item_var = "正式阶段刺激图片/Item名",
    filter_rt = TRUE
  ),
  EmotionStoop = list(
    fun      = analyze_flanker_stroop,
    rt_var   = "相对时间(秒)",
    rt_max   = 2.5,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    item_var = "正式阶段刺激图片/Item名",
    filter_rt = TRUE
  ),
  SST = list(
    fun      = analyze_sst,
    rt_var   = "相对时间(秒)",
    rt_max   = 2.0,               
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    ssd_var  = "SSRT",             
    filter_rt = FALSE              
  ),
  DCCS = list(
    fun          = analyze_switch_cost,
    rt_var       = "相对时间(秒)",
    rt_max       = 2.5,
    min_prop     = 0.5,
    ans_var      = "正式阶段正确答案",
    resp_var     = "正式阶段被试按键",
    switch_var   = "切换类型",
    switch_level = "switch",
    repeat_level = "repeat",
    item_var     = "正式阶段刺激图片/Item名",
    order_var    = "游戏序号",
    mixed_from   = 22,
    filter_rt = TRUE
  ),
  DT = list(
    fun          = analyze_switch_cost,
    rt_var       = "相对时间(秒)",
    rt_max       = 2.5,
    min_prop     = 0.5,
    ans_var      = "正式阶段正确答案",
    resp_var     = "正式阶段被试按键",
    switch_var   = "切换类型",
    switch_level = "switch",
    repeat_level = "repeat",
    item_var     = "正式阶段刺激图片/Item名",
    order_var    = "游戏序号",
    mixed_from   = 66,
    filter_rt = TRUE
  ),
  EmotionSwitch = list(
    fun          = analyze_switch_cost,
    rt_var       = "相对时间(秒)",
    rt_max       = 2.5,
    min_prop     = 0.5,
    ans_var      = "正式阶段正确答案",
    resp_var     = "正式阶段被试按键",
    switch_var   = "切换类型",
    switch_level = "switch",
    repeat_level = "repeat",
    item_var     = "正式阶段刺激图片/Item名",
    order_var    = "游戏序号",
    mixed_from   = 66,
    filter_rt = TRUE
  ),
  GNG = list(
    fun      = analyze_gonogo_cpt,           
    rt_var   = "相对时间(秒)",
    rt_max   = 3,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    filter_rt = FALSE
  ),
  CPT = list(
    fun      = analyze_gonogo_cpt,           
    rt_var   = "相对时间(秒)",
    rt_max   = 3,
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    filter_rt = FALSE
  ),
  ZYST = list(
    fun       = analyze_zyst,
    rt_var    = "相对时间(秒)",          
    min_prop  = 0.5,                     
    ans_var   = "正式阶段正确答案",
    resp_var  = "正式阶段被试按键",
    order_var = "游戏序号",
    filter_rt = FALSE
  ),
  FZSS = list(
    fun      = analyze_fzss,
    rt_var   = "相对时间(秒)",
    rt_max   = 2.0,                
    min_prop = 0.5,
    ans_var  = "正式阶段正确答案",
    resp_var = "正式阶段被试按键",
    filter_rt = TRUE                
  )
)

modalities <- c("number","spatial","emotion")
for (m in modalities) {
  task_config[[paste0("oneback_", m)]] <- task_config$oneback
  task_config[[paste0("twoback_", m)]] <- task_config$twoback
}

######within subject#####
process_subject <- function(file_path, task_config) {
  subject_id <- tools::file_path_sans_ext(basename(file_path))
  sheets     <- excel_sheets(file_path)
  
  res_list <- map(sheets, function(sh) {
    
    sh_low <- tolower(sh)
    
    # ---- 1back ----
    if (grepl("1back", sh_low)) {
      modality <- stringr::str_extract(sh_low, 
                                       "(number|spatial|emotion)")
      if (is.na(modality)) modality <- "raw"
      task_name <- paste0("oneback_", modality)
      
      # ---- 2back ----
    } else if (grepl("2back", sh_low)) {
      modality <- stringr::str_extract(sh_low, 
                                       "(number|spatial|emotion)")
      if (is.na(modality)) modality <- "raw"
      task_name <- paste0("twoback_", modality)
      
    } else {
      task_name <- sh     
    }
    
    # miss Task config → skip
    if (!task_name %in% names(task_config)) {
      message("skip：sheet=", sh)
      return(NULL)
    }
    
    cfg <- task_config[[task_name]]
    fun <- cfg$fun
    
    df <- read_excel(file_path, sheet = sh)
    fun(df, cfg = cfg, subject_id = subject_id, task_name = task_name)
  })
  
  names(res_list) <- sheets
  res_list
}

remove_outliers_3sd <- function(df, metric) {
  m <- mean(df[[metric]], na.rm = TRUE)
  s <- sd(df[[metric]],   na.rm = TRUE)
  
  df %>%
    mutate(z = (.[[metric]] - m) / s) %>%
    filter(abs(z) <= 3)
}

##in order to match the MRIdata and the demographic information
make_sub_id <- function(raw_subject) {
  base <- raw_subject

  parts <- strsplit(base, "_")[[1]]
  idx_cn <- which(stringr::str_detect(parts, "[\u4e00-\u9fa5]"))
  if (length(idx_cn) > 0) {
    parts <- parts[seq_len(idx_cn[1] - 1)]
  }
  
  core <- paste0(parts, collapse = "")
  paste0("sub-", core)
}

######################## For all subjects ####################################
# data path
data_dir <- "/Users/tanlirou/Documents/EFNY/data/Behave/cibr_app_data"
files <- list.files(
  data_dir,
  pattern   = "GameData\\.xlsx$",
  full.names = TRUE
)

# length(files)
# head(files)

# process_subject for each subject
all_res <- files %>%
  set_names() %>%   
  purrr::map(~ process_subject(.x, task_config))

long_res <- all_res %>%
  purrr::map(purrr::compact) %>%   # delete the Null task in each subject
  purrr::map(dplyr::bind_rows) %>% # combine all tasks
  dplyr::bind_rows()               # combine all subjects

dplyr::glimpse(long_res)


long_res <- long_res %>%
  dplyr::mutate(
    subject_raw = subject,
    subject     = vapply(subject_raw, make_sub_id, character(1))
  )

long_res %>% dplyr::select(subject_raw, subject, task) %>% head()

tasks <- sort(unique(long_res$task))
tasks

wide_list <- purrr::map(tasks, function(tk) {
  df_tk <- long_res %>%
    dplyr::filter(task == tk) %>%
    dplyr::select(-task)  
  
  if ("subject_raw" %in% names(df_tk)) {
    df_tk <- df_tk %>% dplyr::select(-subject_raw)
  }
  
  df_tk <- df_tk %>%
    dplyr::group_by(subject) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup()
  
  df_tk <- df_tk %>%
    dplyr::select(
      subject,
      dplyr::where(~ !all(is.na(.x)))
    )
  
  df_tk %>%
    dplyr::rename_with(
      ~ ifelse(.x == "subject", "subject", paste0(tk, "_", .x)),
      dplyr::everything()
    )
})

wide_list <- wide_list[purrr::map_lgl(wide_list, ~ nrow(.x) > 0)]

wide_res <- purrr::reduce(wide_list, dplyr::full_join, by = "subject")

dplyr::glimpse(wide_res)

# save
out_path <- "/Users/tanlirou/Documents/EFNY/data/Behave/results/THU_app_results.csv"
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.csv(wide_res, out_path, row.names = FALSE)

cat("save to", out_path, "\n")

