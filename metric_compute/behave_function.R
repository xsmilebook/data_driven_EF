library(dplyr)
library(purrr)
library(readxl)
library(rlang)
library(stringr)
library(tidyr)
library(tibble)

`%||%` <- function(a, b) if (is.null(a)) b else a
################ General rule ################
prepare_trials <- function(df, cfg, subject_id, task_name) {
  # Number of raw trials
  n_raw <- nrow(df)
  
  # Accuracy: compute correct_trial
  
  if (!is.null(cfg$ans_var) && !is.null(cfg$resp_var) &&
      cfg$ans_var %in% names(df) && cfg$resp_var %in% names(df)) {
    
    df <- df %>%
      dplyr::mutate(
        ans  = .data[[cfg$ans_var]],    # correct answer
        resp = .data[[cfg$resp_var]],   # response
        correct_trial = dplyr::case_when(
          # if ans is NA, mark this trial as NA
          is.na(ans) ~ NA,
          
          # has answer but no response → incorrect
          !is.na(ans) & is.na(resp) ~ FALSE,
          # exact match between ans and resp → correct
          as.character(ans) == as.character(resp) ~ TRUE,
          
          # all other cases → incorrect
          TRUE ~ FALSE
        )
      ) %>%
      dplyr::select(-ans, -resp)   
    
  } else {
    # If answer/response columns missing, set correct_trial to NA
    df$correct_trial <- NA
  }
  
  # RT preprocessing: threshold + 3SD trimming
  # Whether to filter by RT (default TRUE)
  filter_rt <- cfg$filter_rt %||% TRUE
  
  if (!is.null(cfg$rt_var) && cfg$rt_var %in% names(df)) {
    
    df <- df %>%
      dplyr::mutate(
        # Convert RT column to numeric
        rt  = as.numeric(.data[[cfg$rt_var]]),
        # Preserve original order for later re-merging
        .trial_order = dplyr::row_number()
      )
    
    if (isTRUE(filter_rt)) {
      
      
      # Split into "has RT" vs "no RT"
      
      has_rt    <- !is.na(rt)
      df_rt     <- df[has_rt, , drop = FALSE]  
      df_no_rt  <- df[!has_rt, , drop = FALSE] 
      
      n_rt_na     <- sum(!has_rt)     
      n_rt_before <- nrow(df_rt)        
      
      
      # Threshold-based RT filtering
      df_rt <- df_rt %>%
        dplyr::filter(rt >= 0.2)         
      
      if (!is.null(cfg$rt_max)) {
        df_rt <- df_rt %>%
          dplyr::filter(rt <= cfg$rt_max)  
      }
      
      
      # 3SD trimming on remaining RT
      
      if (nrow(df_rt) > 0) {
        m_rt <- mean(df_rt$rt, na.rm = TRUE)
        s_rt <- stats::sd(df_rt$rt,  na.rm = TRUE)
        
        # Only trim if sd is positive and finite
        if (is.finite(s_rt) && s_rt > 0) {
          df_rt <- df_rt %>%
            dplyr::mutate(
              z = (rt - m_rt) / s_rt 
            ) %>%
            dplyr::filter(abs(z) <= 3) %>%   
            dplyr::select(-z)                
        }
      }
      # Number of RT trials after threshold + 3SD filtering
      n_rt_after   <- nrow(df_rt)
      # Total number of RT trials deleted (by threshold + 3SD)
      n_rt_deleted <- n_rt_before - n_rt_after
      
      # Merge back with "no RT" trials and restore original order
      
      df <- dplyr::bind_rows(df_rt, df_no_rt) %>%
        dplyr::arrange(.trial_order) %>%
        dplyr::select(-.trial_order)
      
      # Proportion check:(deleted RT + missing RT) must not exceed (1 - min_prop)
      min_prop <- cfg$min_prop %||% 0.5  
      # "problematic" trials = deleted RT trials + NA RT trials
      n_problem <- n_rt_deleted + n_rt_na
      
      if (n_raw > 0 && n_problem > n_raw * (1 - min_prop)) {
        warning(sprintf(
          "被试 %s - 任务 %s：RT 被删 %d 个，RT 为空 %d 个，共占 %.1f%%，不分析该任务",
          subject_id, task_name, n_rt_deleted, n_rt_na,
          100 * n_problem / n_raw
        ))
        return(list(ok = FALSE, df = df, n_raw = n_raw, n_kept = nrow(df)))
      }
      
    } else {
      # Not filtering by RT: just drop the temporary order column
      df <- df %>% dplyr::select(-.trial_order)
    }
    
  } else if (isTRUE(filter_rt)) {
    # If RT filtering requested but rt_var missing, throw error
    stop(sprintf("任务 %s 缺少 rt_var 指定的列", task_name))
  }
  list(
    ok    = TRUE,df = df, n_raw = n_raw, n_kept= nrow(df) )
}

prepare_sst <- function(df, cfg, subject_id, task_name) {
  # Keep only the first 96 trials
  if (nrow(df) >= 96) {
    df <- df[1:96, ]
  }
  
  # Create unified rt column
  rt_var <- cfg$rt_var %||% "相对时间(秒)"
  
  if (rt_var %in% names(df)) {
    df <- df %>%
      mutate(rt = as.numeric(.data[[rt_var]]))
  } else {
    stop("SST：缺少 RT 列 ", rt_var)
  }
  
  #  Extract answer and response columns
  ans_var  <- cfg$ans_var  %||% "正式阶段正确答案"
  resp_var <- cfg$resp_var %||% "正式阶段被试按键"
  
  if (ans_var %in% names(df) && resp_var %in% names(df)) {
    df <- df %>%
      mutate(
        correct_trial = as.character(.data[[ans_var]]) ==
          as.character(.data[[resp_var]])
      )
  } else {
    df$correct_trial <- NA
  }
  
  list(ok = TRUE, df = df, n_raw = 96, n_kept = nrow(df))
}



######################## N-back ##############################
compute_dprime <- function(hit_rate, fa_rate) {
  eps <- 0.5 / 100  # avoid 0 and 1
  hit_rate <- pmin(pmax(hit_rate, eps), 1 - eps)
  fa_rate  <- pmin(pmax(fa_rate,  eps), 1 - eps)
  qnorm(hit_rate) - qnorm(fa_rate)
}

analyze_1back <- function(df, cfg, subject_id, task_name) {
  # 1) Preprocess trials 
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  # trial_type = target / nontarget based on 1-back stimulus match
  
  # stimulus column name // 刺激列名
  stim_var <- cfg$stim_var %||% "正式阶段刺激图片/Item名"
  
  if (!stim_var %in% names(df)) {
    stop("1-back：找不到刺激列 ", stim_var)
  }
  
  df <- df %>%
    dplyr::mutate(
      stim      = .data[[stim_var]],           
      stim_lag1 = dplyr::lag(stim, 1),         
      trial_type = dplyr::case_when(
        !is.na(stim) & !is.na(stim_lag1) & stim == stim_lag1 ~ "target",
        !is.na(stim) ~ "nontarget",
        TRUE ~ NA_character_
      )
    ) %>%
    dplyr::select(-stim_lag1)
  
  # check  trial_type 
  if (!"trial_type" %in% names(df)) {
    stop("1-back：需要 trial_type 列 (target / nontarget)")
  }
  
  # Compute hit rate and false alarm rate
  hit_rate <- mean(df$correct_trial[df$trial_type == "target"],     na.rm = TRUE)
  fa_rate  <- mean(!df$correct_trial[df$trial_type == "nontarget"], na.rm = TRUE)
  df_valid <- df %>% dplyr::filter(!is.na(trial_type))
  # Summary
  tibble::tibble(
    subject  = subject_id,
    task     = task_name,
    n_back   = 1L,
    n_raw    = prep$n_raw,
    n_kept   = nrow(df_valid),
    mean_rt  = mean(df_valid$rt, na.rm = TRUE),
    acc      = mean(df_valid$correct_trial, na.rm = TRUE),
    hit_rate = mean(df_valid$correct_trial[df_valid$trial_type == "target"],     na.rm = TRUE),
    fa_rate  = mean(!df_valid$correct_trial[df_valid$trial_type == "nontarget"], na.rm = TRUE),
    dprime   = compute_dprime(hit_rate, fa_rate)
  )
}

analyze_2back <- function(df, cfg, subject_id, task_name) {
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  stim_var <- cfg$stim_var %||% "正式阶段刺激图片/Item名"
  
  if (!stim_var %in% names(df)) {
    stop("2-back：找不到刺激列 ", stim_var)
  }
  
  df <- df %>%
    mutate(
      stim      = .data[[stim_var]],
      stim_lag2 = lag(stim, 2),
      trial_type = case_when(
        !is.na(stim) & !is.na(stim_lag2) & stim == stim_lag2 ~ "target",
        !is.na(stim)                                          ~ "nontarget",
        TRUE                                                   ~ NA_character_
      )
    ) %>%
    select(-stim_lag2)
  
  df_valid <- df %>% filter(!is.na(trial_type))
  
  hit_rate <- mean(df_valid$correct_trial[df_valid$trial_type == "target"], na.rm = TRUE)
  fa_rate  <- mean(!df_valid$correct_trial[df_valid$trial_type == "nontarget"], na.rm = TRUE)
  
  tibble(
    subject  = subject_id,
    task     = task_name,
    n_raw    = prep$n_raw,
    n_kept   = nrow(df_valid),
    mean_rt  = mean(df_valid$rt, na.rm = TRUE),
    acc      = mean(df_valid$correct_trial, na.rm = TRUE),
    hit_rate = hit_rate,
    fa_rate  = fa_rate,
    dprime   = compute_dprime(hit_rate, fa_rate)
  )
}


######################## KeepTrack ##############################
analyze_keeptrack <- function(df, cfg, subject_id, task_name) {
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  tibble(
    subject = subject_id,
    task    = task_name,
    n_raw   = prep$n_raw,
    n_kept  = prep$n_kept,
    mean_rt = mean(df$rt, na.rm = TRUE),
    acc     = mean(df$correct_trial, na.rm = TRUE)
  )
}

######################## FLANKER ColorStroop EmotionStroop ##############################
add_flanker_condition <- function(df, item_var = "正式阶段刺激图片/Item名") {
  if (!item_var %in% names(df)) {
    stop("Flanker：找不到列 ", item_var)
  }
  
  df %>%
    dplyr::mutate(
      # cond: congruent vs incongruent based on item suffix
      cond = dplyr::case_when(
        stringr::str_detect(.data[[item_var]], "LL$|RR$") ~ "congruent",
        stringr::str_detect(.data[[item_var]], "LR$|RL$") ~ "incongruent", 
        TRUE ~ NA_character_                               
      )
    )
}

add_colorstroop_condition <- function(df, item_var = "正式阶段刺激图片/Item名") {
  if (!item_var %in% names(df)) {
    stop("ColorStroop：找不到列 ", item_var)
  }
  
  df %>%
    # Split item name into components: Pic, PicColor, Text, TextColor
    tidyr::separate(
      col    = tidyselect::all_of(item_var),
      into   = c("Pic", "PicColor", "Text", "TextColor"),
      sep    = "_",
      remove = FALSE
    ) %>%
    dplyr::mutate(
      # If picture color equals text color → congruent, else incongruent
      cond = dplyr::if_else(PicColor == TextColor, "congruent", "incongruent")
    )
}

# Add condition for Emotion Stroop task
add_emotionstroop_condition <- function(df,
                                        item_var = "正式阶段刺激图片/Item名") {
  if (!item_var %in% names(df)) {
    stop("EmotionStroop：找不到列 ", item_var)
  }
  
  df %>%
    dplyr::mutate(
      # Extract numbers from item name
      stim_num = readr::parse_number(as.character(.data[["正式阶段刺激图片/Item名"]])),
      
      # multiples of 4 are congruent, others incongruent
      cond = dplyr::if_else(
        !is.na(stim_num) & stim_num %% 4 == 0,
        "congruent",     
        "incongruent"     
      )
    )
}

analyze_flanker_stroop <- function(df, cfg, subject_id, task_name) {
  # Preprocess 
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  item_var <- cfg$item_var %||% "正式阶段刺激图片/Item名"
  
  # Add condition (congruent / incongruent) by task
  if (task_name == "FLANKER") {
    df <- add_flanker_condition(df, item_var = item_var)
  } else if (task_name == "ColorStroop") {
    df <- add_colorstroop_condition(df, item_var = item_var)
  } else if (task_name == "EmotionStroop") {
    df <- add_emotionstroop_condition(df, item_var = item_var)
  }
  
  if (!"cond" %in% names(df)) {
    stop(task_name, "：需要 cond 列 (congruent / incongruent)")
  }
  
  # Summary by condition
  summary <- df %>%
    dplyr::group_by(cond) %>%
    dplyr::summarise(
      mean_rt = mean(rt, na.rm = TRUE),            
      acc     = mean(correct_trial, na.rm = TRUE), 
      .groups = "drop"
    )
  
  # avoid numeric(0) when a condition is missing
  cong_rt <- if ("congruent" %in% summary$cond) {
    summary$mean_rt[summary$cond == "congruent"]
  } else NA_real_
  
  incong_rt <- if ("incongruent" %in% summary$cond) {
    summary$mean_rt[summary$cond == "incongruent"]
  } else NA_real_
  
  cong_acc <- if ("congruent" %in% summary$cond) {
    summary$acc[summary$cond == "congruent"]
  } else NA_real_
  
  incong_acc <- if ("incongruent" %in% summary$cond) {
    summary$acc[summary$cond == "incongruent"]
  } else NA_real_
  
  # summary 
  tibble::tibble(
    subject    = subject_id, 
    task       = task_name,
    n_raw      = prep$n_raw, 
    n_kept     = prep$n_kept, 
    cong_rt    = cong_rt, 
    incong_rt  = incong_rt, 
    cong_acc   = cong_acc, 
    incong_acc = incong_acc,
    diff_rt    = incong_rt - cong_rt, 
    diff_acc   = incong_acc - cong_acc 
  )
}

######################## DCCS DT EmotionSwitch ##############################
add_switch_condition <- function(df,
                                 task_name,
                                 ans_var    = "正式阶段正确答案",
                                 item_var   = "正式阶段刺激图片/Item名",
                                 order_var  = "游戏序号",
                                 switch_var = "切换类型") {
  # Sort by trial order if order_var exists
  if (order_var %in% names(df)) {
    df <- df %>% dplyr::arrange(.data[[order_var]])
  }
  
  # DCCS: use the first letter of item_var as the task cue
  if (task_name == "DCCS") {
    if (!item_var %in% names(df)) {
      stop("DCCS：找不到刺激列 ", item_var)
    }
    
    df <- df %>%
      dplyr::mutate(
        cue      = substr(.data[[item_var]], 1, 1),   # First char of stimulus
        prev_cue = dplyr::lag(cue),  # Previous trial cue
        !!switch_var := dplyr::case_when(
          is.na(prev_cue) ~ NA_character_,  # First trial → NA
          cue == prev_cue ~ "repeat", # Same cue → repeat
          TRUE  ~ "switch" # Different cue → switch
        )
      ) %>%
      dplyr::select(-cue, -prev_cue)
    
    return(df)
  }
  
  # DT: detect "T/t" in item_var → TN; otherwise CN
  if (task_name == "DT") {
    if (!item_var %in% names(df)) {
      stop("DT：找不到刺激列 ", item_var)
    }
    
    df <- df %>%
      dplyr::mutate(
        task_block = dplyr::case_when(
          stringr::str_detect(.data[[item_var]], "[Tt]") ~ "TN",  # Contains T → TN
          TRUE                                           ~ "CN"   # Otherwise → CN
        ),
        prev_block = dplyr::lag(task_block),
        !!switch_var := dplyr::case_when(
          is.na(prev_block)        ~ NA_character_,     # First trial
          task_block == prev_block ~ "repeat",          # Same block
          TRUE ~ "switch"           # Different block
        )
      ) %>%
      dplyr::select(-task_block, -prev_block)
    
    return(df)
  }

  # EmotionSwitch: 1–4 → emotion; 5–8 → gender
  if (task_name == "EmotionSwitch") {
    if (!item_var %in% names(df)) {
      stop("EmotionSwitch：找不到刺激列 ", item_var)
    }
    
    df <- df %>%
      dplyr::mutate(
        stim_num =readr::parse_number(as.character(.data[["正式阶段刺激图片/Item名"]])),  # Extract 1–8
        task_block = dplyr::case_when(
          stim_num %in% 1:4 ~ "emotion",   # Emotion task
          stim_num %in% 5:8 ~ "gender",   # Gender task
          TRUE              ~ NA_character_
        ),
        prev_block = dplyr::lag(task_block),
        !!switch_var := dplyr::case_when(
          is.na(prev_block)        ~ NA_character_,
          task_block == prev_block ~ "repeat",
          TRUE                     ~ "switch"
        )
      ) %>%
      dplyr::select(-stim_num, -task_block, -prev_block)
    
    return(df)
  }

  warning("add_switch_condition：未定义的 switch 任务类型：", task_name)
  df
}

analyze_switch_cost <- function(df, cfg, subject_id, task_name) {
  # General preprocessing (RT filtering + correct_trial)
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  sw_var    <- cfg$switch_var %||% "切换类型"
  order_var <- cfg$order_var  %||% "游戏序号"
  
  # Label switch / repeat based on task-specific rule
  df <- add_switch_condition(
    df,
    task_name = task_name,
    ans_var   = cfg$ans_var %||% "正式阶段正确答案",
    item_var  = cfg$item_var %||% "正式阶段刺激图片/Item名",
    order_var = order_var,
    switch_var = sw_var
  )
  
  # Drop first trial of each block
  df <- df %>% dplyr::filter(!is.na(.data[[sw_var]]))
  
  # Keep only mixed-block trials
  if (!is.null(cfg$mixed_from) && order_var %in% names(df)) {
    df <- df %>% dplyr::filter(.data[[order_var]] >= cfg$mixed_from)
  }
  if (!is.null(cfg$mixed_to) && order_var %in% names(df)) {
    df <- df %>% dplyr::filter(.data[[order_var]] <= cfg$mixed_to)
  }
  
  # If no trials left, return NA summary
  if (nrow(df) == 0) {
    warning("被试 ", subject_id, " - 任务 ", task_name,
            "：mixed block 里没有有效 trial，返回 NA")
    return(
      tibble::tibble(
        subject        = subject_id,
        task           = task_name,
        n_raw          = prep$n_raw,
        n_kept         = prep$n_kept,
        rt_switch      = NA_real_,
        rt_repeat      = NA_real_,
        acc_switch     = NA_real_,
        acc_repeat     = NA_real_,
        switch_cost_rt = NA_real_,
        switch_cost_acc= NA_real_
      )
    )
  }
  
  # Summary by switch / repeat
  summary <- df %>%
    dplyr::group_by(.data[[sw_var]]) %>%
    dplyr::summarise(
      mean_rt = mean(rt, na.rm = TRUE),
      acc     = mean(correct_trial, na.rm = TRUE),
      .groups = "drop"
    )
  
  sw_level <- cfg$switch_level %||% "switch"
  rp_level <- cfg$repeat_level %||% "repeat"
  
  rt_switch  <- if (sw_level %in% summary[[sw_var]]) {
    summary$mean_rt[summary[[sw_var]] == sw_level]
  } else NA_real_
  
  rt_repeat  <- if (rp_level %in% summary[[sw_var]]) {
    summary$mean_rt[summary[[sw_var]] == rp_level]
  } else NA_real_
  
  acc_switch <- if (sw_level %in% summary[[sw_var]]) {
    summary$acc[summary[[sw_var]] == sw_level]
  } else NA_real_
  
  acc_repeat <- if (rp_level %in% summary[[sw_var]]) {
    summary$acc[summary[[sw_var]] == rp_level]
  } else NA_real_
  
  tibble::tibble(
    subject        = subject_id,
    task           = task_name,
    n_raw          = prep$n_raw,
    n_kept         = prep$n_kept,               
    rt_switch      = rt_switch,
    rt_repeat      = rt_repeat,
    acc_switch     = acc_switch,
    acc_repeat     = acc_repeat,
    switch_cost_rt  = rt_switch - rt_repeat,    # RT cost: switch - repeat
    switch_cost_acc = acc_switch - acc_repeat   # ACC cost: switch - repeat
  )
}


######################## CPT GNG ##############################
add_gonogo_cpt_condition <- function(df, ans_var = "正式阶段正确答案") {
  if (!ans_var %in% names(df)) {
    stop("GoNoGo：找不到列 ", ans_var)
  }
  
  df %>%
    dplyr::mutate(
      # Convert answer column to character, then map to go / nogo
      ans_chr = as.character(.data[[ans_var]]),
      trial_type = dplyr::case_when(
        ans_chr %in% c("TRUE","True","true","1")   ~ "go",    # go trial
        ans_chr %in% c("FALSE","False","false","0") ~ "nogo",  # nogo trial
        TRUE ~ NA_character_  
      )
    ) %>%
    dplyr::select(-ans_chr)   
}

analyze_gonogo_cpt <- function(df, cfg, subject_id, task_name) {
  # Preprocess without RT filtering
  cfg_nofilter <- cfg
  cfg_nofilter$filter_rt <- FALSE
  
  prep <- prepare_trials(df, cfg_nofilter, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  ans_var  <- cfg$ans_var  %||% "正式阶段正确答案"
  rt_var   <- cfg$rt_var   %||% "相对时间(秒)"
  rt_max   <- cfg$rt_max   %||% Inf
  min_prop <- cfg$min_prop %||% 0.5  
  
  
  # Determine trial_type = go / nogo
  
  df <- add_gonogo_cpt_condition(df, ans_var = ans_var)
  
  if (!"trial_type" %in% names(df)) {
    stop("GoNoGo：需要 trial_type 列 (go / nogo)")
  }
  
  go_trials_raw   <- df %>% dplyr::filter(trial_type == "go")
  nogo_trials_all <- df %>% dplyr::filter(trial_type == "nogo")
  
  # RT threshold filtering for go trials
  
  if (nrow(go_trials_raw) > 0 && rt_var %in% names(df)) {
    
    go_trials_rt <- go_trials_raw %>%
      dplyr::mutate(rt = as.numeric(.data[[rt_var]]))

    has_rt <- !is.na(go_trials_rt$rt)
    go_with_rt  <- go_trials_rt[has_rt, , drop = FALSE]
    n_rt_na     <- sum(!has_rt)
    n_rt_before <- nrow(go_with_rt)
    
    # RT filtering
    go_with_rt <- go_with_rt %>%
      dplyr::filter(rt >= 0.2) %>% 
      dplyr::filter(rt <= rt_max)

    # 3SD trimming for go trials
    
    if (nrow(go_with_rt) > 0) {
      m_rt <- mean(go_with_rt$rt, na.rm = TRUE)
      s_rt <- sd(go_with_rt$rt,  na.rm = TRUE)
      
      if (is.finite(s_rt) && s_rt > 0) {
        go_with_rt <- go_with_rt %>%
          dplyr::mutate(z = (rt - m_rt) / s_rt) %>%
          dplyr::filter(abs(z) <= 3) %>%
          dplyr::select(-z)
      }
    }
    
    n_rt_after   <- nrow(go_with_rt)
    n_rt_deleted <- n_rt_before - n_rt_after
    
    
    # Check if at least min_prop go trials remain
    
    n_go_all <- nrow(go_trials_raw)
    n_problem <- n_rt_deleted + n_rt_na
    
    if (n_go_all > 0 && n_problem > n_go_all * (1 - min_prop)) {
      warning(sprintf(
        "被试 %s - 任务 %s：go trial 中 RT 被删 %d 个，RT 为空 %d 个，共 %.1f%%，跳过分析",
        subject_id, task_name, n_rt_deleted, n_rt_na,
        100 * n_problem / n_go_all
      ))
      return(NULL)
    }
    
    go_trials_valid <- go_with_rt
    
  } else {
    go_trials_valid <- go_trials_raw[0, , drop = FALSE]
  }
  
  
  # Compute go_acc, nogo_acc, go_rt
  
  go_acc <- if (nrow(go_trials_valid) > 0) {
    mean(go_trials_valid$correct_trial, na.rm = TRUE)
  } else NA_real_
  
  nogo_acc <- if (nrow(nogo_trials_all) > 0) {
    mean(nogo_trials_all$correct_trial, na.rm = TRUE)
  } else NA_real_
  
  go_rt <- if (nrow(go_trials_valid) > 0) {
    mean(go_trials_valid$rt, na.rm = TRUE)
  } else NA_real_
  
  
  # Summary
  
  tibble::tibble(
    subject  = subject_id, 
    task     = task_name,
    n_raw    = prep$n_raw, 
    n_kept   = nrow(go_trials_valid) + nrow(nogo_trials_all),
    go_rt    = go_rt,   
    go_acc   = go_acc, 
    nogo_acc = nogo_acc    
  )
}
######################## SST ##############################
analyze_sst <- function(df, cfg, subject_id, task_name) {
 
  # Use SST-specific prepare function
  prep <- prepare_sst(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  ans_var  <- cfg$ans_var  %||% "正式阶段正确答案"
  resp_var <- cfg$resp_var %||% "正式阶段被试按键"
  ssd_var  <- cfg$ssd_var  %||% "SSRT"      # SSD 列名
  rt_max   <- cfg$rt_max   %||% Inf         # go RT 上限
  min_prop <- cfg$min_prop %||% 0.5         # 至少保留 50% 的正确 go trial
  
 
  # Check SSD column for stop trials
  if (!ssd_var %in% names(df)) {
    message("⚠️ SST：未找到 SSD 列 ", ssd_var,
            "，无法计算 SSRT（被试 = ", subject_id, ")")
    return(
      tibble::tibble(
        subject    = subject_id,
        task       = task_name,
        n_raw      = prep$n_raw,
        n_kept     = prep$n_kept,
        mean_go_rt = NA_real_,
        stop_acc   = NA_real_,
        mean_ssd   = NA_real_,
        ssrt       = NA_real_
      )
    )
  }
  
 
  # Mark stop vs. go trials + define correctness
 
  df <- df %>%
    dplyr::mutate(
      # SSD numeric column 
      ssd = suppressWarnings(as.numeric(.data[[ssd_var]])),
      
      # stop trial: SSD non-missing
      is_stop = !is.na(ssd),
      
      # correctness:
      #  - go trial: use correct_trial from prepare_sst
      #  - stop trial: correct if NO response (resp_var = NA)
      correct = dplyr::case_when(
        !is_stop ~ as.integer(correct_trial),
        is_stop  ~ as.integer(is.na(.data[[resp_var]]))
      )
    )
  
  # Separate stop / go trials
  # 分开 stop trial 和 go trial
  stop_trials <- df %>% dplyr::filter(is_stop)
  go_trials   <- df %>% dplyr::filter(!is_stop)
  
  # Stop accuracy & mean SSD
  stop_acc <- mean(stop_trials$correct, na.rm = TRUE)
  mean_ssd <- mean(stop_trials$ssd,     na.rm = TRUE)
  
 
  # Apply RT threshold + 3SD trimming + 50% rule on correct go trials
 
  # Correct go trials only
  go_corr <- go_trials %>% dplyr::filter(correct == 1)
  
  if (nrow(go_corr) == 0) {
    warning("被试 ", subject_id, " - 任务 ", task_name,
            "：没有正确的 go trial，无法计算 SSRT")
    return(
      tibble::tibble(
        subject    = subject_id,
        task       = task_name,
        n_raw      = prep$n_raw,
        n_kept     = prep$n_kept,
        mean_go_rt = NA_real_,
        stop_acc   = stop_acc,
        mean_ssd   = mean_ssd,
        ssrt       = NA_real_
      )
    )
  }
  
  # RT threshold filtering
  has_rt      <- !is.na(go_corr$rt)
  go_corr_rt  <- go_corr[has_rt, , drop = FALSE]
  n_rt_na     <- sum(!has_rt)  
  n_rt_before <- nrow(go_corr_rt)    
  
  go_corr_rt <- go_corr_rt %>%
    dplyr::filter(rt >= 0.2)
  if (is.finite(rt_max)) {
    go_corr_rt <- go_corr_rt %>%
      dplyr::filter(rt <= rt_max)
  }
  
  # 3SD trimming on remaining correct go trials
  if (nrow(go_corr_rt) > 0) {
    m_rt <- mean(go_corr_rt$rt, na.rm = TRUE)
    s_rt <- stats::sd(go_corr_rt$rt,  na.rm = TRUE)
    
    if (is.finite(s_rt) && s_rt > 0) {
      go_corr_rt <- go_corr_rt %>%
        dplyr::mutate(z = (rt - m_rt) / s_rt) %>%
        dplyr::filter(abs(z) <= 3) %>%
        dplyr::select(-z)
    }
  }
  
  n_rt_after   <- nrow(go_corr_rt)
  n_rt_deleted <- n_rt_before - n_rt_after  
  # problematic correct go trials = deleted + missing RT
  n_problem <- n_rt_deleted + n_rt_na
  n_corr_all <- nrow(go_corr)            
  
  if (n_corr_all > 0 && n_problem > n_corr_all * (1 - min_prop)) {
    warning(sprintf(
      "被试 %s - 任务 %s：正确 go trial 中 RT 被删 %d 个，RT 为空 %d 个，共占 %.1f%%，SSRT 不分析",
      subject_id, task_name, n_rt_deleted, n_rt_na,
      100 * n_problem / n_corr_all
    ))
    return(
      tibble::tibble(
        subject    = subject_id,
        task       = task_name,
        n_raw      = prep$n_raw,
        n_kept     = prep$n_kept,
        mean_go_rt = NA_real_,
        stop_acc   = stop_acc,
        mean_ssd   = mean_ssd,
        ssrt       = NA_real_
      )
    )
  }
  
  go_trials_valid_correct <- go_corr_rt

  if (nrow(go_trials_valid_correct) == 0) {
    warning("被试 ", subject_id, " - 任务 ", task_name,
            "：经过 RT 过滤和 3SD 剪裁后，没有正确 go trial，SSRT 为 NA")
    return(
      tibble::tibble(
        subject    = subject_id,
        task       = task_name,
        n_raw      = prep$n_raw,
        n_kept     = prep$n_kept,
        mean_go_rt = NA_real_,
        stop_acc   = stop_acc,
        mean_ssd   = mean_ssd,
        ssrt       = NA_real_
      )
    )
  }
  
 
  # Integration / percentile method for SSRT
  if (stop_acc > 0 && stop_acc < 1) {
    sorted_go_rt <- sort(go_trials_valid_correct$rt)
    p   <- 1 - stop_acc
    n   <- length(sorted_go_rt)
    idx <- max(1, min(n, floor(n * p)))   
    
    go_rt_p <- sorted_go_rt[idx]
    ssrt    <- go_rt_p - mean_ssd
  } else {
    ssrt <- NA_real_
  }
  # Mean go RT: based on valid correct go trials
  mean_go_rt <- mean(go_trials_valid_correct$rt, na.rm = TRUE)
  
 
  # Summary
  tibble::tibble(
    subject    = subject_id,
    task       = task_name,
    n_raw      = prep$n_raw,
    n_kept     = prep$n_kept,
    mean_go_rt = mean_go_rt,
    stop_acc   = stop_acc,
    mean_ssd   = mean_ssd,
    ssrt       = ssrt
  )
}


######################## ZYST ##############################
extract_trial_info_zyst <- function(df, order_var = "游戏序号") {
  if (!order_var %in% names(df)) {
    stop("ZYST：找不到列 ", order_var)
  }
  
  # Ensure character before extraction
  x <- as.character(df[[order_var]])
  
  # Match pattern like "12-0" or "12 - 0" at the beginning of the string
  mat <- stringr::str_match(x, "^(\\d+)\\s*-\\s*(\\d+)")
  
  # mat[,2] = trial index, mat[,3] = subtrial index
  df$trial    <- as.integer(mat[, 2])
  df$subtrial <- as.integer(mat[, 3])
  
  df
}

analyze_zyst <- function(df, cfg, subject_id, task_name) {
  # General prepare: create correct_trial; no RT filtering
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  # Add trial / subtrial indices
  order_var <- cfg$order_var %||% "游戏序号"
  df <- extract_trial_info_zyst(df, order_var = order_var)
  
  # Minimum number of valid responses (completion check)
  resp_var <- cfg$resp_var %||% "正式阶段被试按键"
  min_resp <- cfg$min_resp %||% 128
  
  n_valid_resp <- sum(!is.na(df[[resp_var]]))
  if (n_valid_resp < min_resp) {
    warning(sprintf(
      "被试 %s - 任务 %s：有效按键只有 %d (< %d)，跳过该任务",
      subject_id, task_name, n_valid_resp, min_resp
    ))
    return(NULL)
  }
  
  # Compute T0 / T1 accuracy
  t0_list <- c()
  t1_list <- c()
  t1_given_t0_correct <- c()
  
  for (trial_id in sort(unique(df$trial))) {
    group <- df[df$trial == trial_id, ]
    
    # Require both subtrial 0 and 1
    if (!all(c(0, 1) %in% group$subtrial)) next
    
    t0 <- group[group$subtrial == 0, ] %>% dplyr::slice(1)
    t1 <- group[group$subtrial == 1, ] %>% dplyr::slice(1)
    
    # correct_trial may be NA: isTRUE(NA) = FALSE, so NA treated as incorrect (0)
    t0_corr <- as.integer(isTRUE(t0$correct_trial))
    t1_corr <- as.integer(isTRUE(t1$correct_trial))
    
    t0_list <- c(t0_list, t0_corr)
    t1_list <- c(t1_list, t1_corr)
    
    # Only include trials where T0 was correct
    if (t0_corr == 1) {
      t1_given_t0_correct <- c(t1_given_t0_correct, t1_corr)
    }
  }
  
  # If no valid trials, return NULL
  if (length(t0_list) == 0 || length(t1_list) == 0) {
    return(NULL)
  }
  
  t0_acc <- mean(t0_list)
  t1_acc <- mean(t1_list)
  
  t1_acc_given_t0_correct <- if (length(t1_given_t0_correct) > 0) {
    mean(t1_given_t0_correct)
  } else {
    NA_real_
  }
  
  tibble::tibble(
    subject  = subject_id,
    task     = task_name,
    n_raw    = prep$n_raw,   
    n_kept   = prep$n_kept,  
    t0_acc   = t0_acc,      
    t1_acc   = t1_acc,       
    t1_acc_given_t0_correct = t1_acc_given_t0_correct  
  )
}


######################## FZSS ##############################
analyze_fzss <- function(df, cfg, subject_id, task_name) {
  prep <- prepare_trials(df, cfg, subject_id, task_name)
  if (!prep$ok) return(NULL)
  df <- prep$df
  
  ans_var  <- cfg$ans_var  %||% "正式阶段正确答案"
  resp_var <- cfg$resp_var %||% "正式阶段被试按键"
  
  df <- df %>%
    dplyr::mutate(
      ans   = as.character(.data[[ans_var]]),
      resp  = as.character(.data[[resp_var]]),
      correct = ans == resp
    )
  
  acc <- mean(df$correct, na.rm = TRUE)
  
  miss_base <- df %>% dplyr::filter(ans == "Right")
  miss_trials <- miss_base %>%
    dplyr::filter(is.na(resp) | resp != "Right")
  miss_rate <- if (nrow(miss_base) > 0) {
    nrow(miss_trials) / nrow(miss_base)
  } else {
    NA_real_
  }
  
  fa_base <- df %>% dplyr::filter(ans == "Left")
  fa_trials <- fa_base %>%
    dplyr::filter(resp == "Right")
  fa_rate <- if (nrow(fa_base) > 0) {
    nrow(fa_trials) / nrow(fa_base)
  } else {
    NA_real_
  }
  
  mean_rt <- df %>%
    dplyr::filter(correct) %>%
    dplyr::summarise(m = mean(rt, na.rm = TRUE)) %>%
    dplyr::pull(m)
  
  tibble::tibble(
    subject       = subject_id,
    task          = task_name,
    n_raw         = prep$n_raw,
    n_kept        = prep$n_kept,
    fzss_acc      = acc,
    fzss_miss_rate= miss_rate,
    fzss_fa_rate  = fa_rate,
    fzss_mean_rt  = mean_rt
  )
}