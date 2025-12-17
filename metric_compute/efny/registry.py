from .metrics import (
    metric_acc,
    metric_contrast_acc,
    metric_contrast_rt,
    metric_dprime,
    metric_rt,
    metric_ssrt,
    metric_switch_cost,
)

KNOWN_METRICS = ['d_prime', 'SSRT', 'ACC', 'Reaction_Time', 'Contrast_ACC', 'Contrast_RT', 'Switch_Cost']
SKIP_TASKS = {'GNG', 'ZYST', 'FZSS'}
REQUIRED_COLUMNS = ['task', 'trial_index', 'subject_id', 'subject_name', 'item', 'answer', 'key', 'rt']

METRIC_FUNCS = {
    'ACC': metric_acc,
    'Reaction_Time': metric_rt,
    'd_prime': metric_dprime,
    'SSRT': metric_ssrt,
    'Contrast_ACC': metric_contrast_acc,
    'Contrast_RT': metric_contrast_rt,
    'Switch_Cost': metric_switch_cost,
}

