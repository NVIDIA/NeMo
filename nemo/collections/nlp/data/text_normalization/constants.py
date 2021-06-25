DECODE_CTX_SIZE = 3
LABEL_PAD_TOKEN_ID = -100

# Task Prefixes
ITN_PREFIX = str(0)
TN_PREFIX = str(1)

# Tagger Labels Prefixes
B_PREFIX = 'B-' # Denote beginning
I_PREFIX = 'I-' # Denote middle
TAGGER_LABELS_PREFIXES = [B_PREFIX, I_PREFIX]

# TAGS
TASK_TAG = 'TASK'
SAME_TAG = 'SAME'
PUNCT_TAG = 'PUNCT'
TRANSFORM_TAG = 'TRANSFORM'
ALL_TAGS = [TASK_TAG, SAME_TAG, PUNCT_TAG, TRANSFORM_TAG]

# ALL_TAG_LABELS
ALL_TAG_LABELS = []
for prefix in TAGGER_LABELS_PREFIXES:
    for tag in ALL_TAGS:
        ALL_TAG_LABELS.append(prefix + tag)
ALL_TAG_LABELS.sort()

# Special Words
SIL_WORD = 'sil'
SELF_WORD = '<self>'
SPECIAL_WORDS = [SIL_WORD, SELF_WORD]

# Greek Letters
GREEK_TO_SPOKEN = {
    'Τ': 'tau', 'Ο': 'omicron', 'Δ': 'delta', 'Η': 'eta', 'Κ': 'kappa',
    'Ι': 'iota', 'Θ': 'theta', 'Α': 'alpha', 'Σ': 'sigma', 'Υ': 'upsilon',
    'Μ': 'mu', 'Ε': 'epsilon', 'Χ': 'chi', 'Π': 'pi', 'Ν': 'nu', 'Λ': 'lambda',
    'Γ': 'gamma', 'Β': 'beta', 'Ρ': 'rho', 'τ': 'tau', 'υ': 'upsilon',
    'μ': 'mu', 'φ': 'phi', 'α': 'alpha', 'λ': 'lambda', 'ι': 'iota',
    'ς': 'sigma', 'ο': 'omicron', 'σ': 'sigma', 'η': 'eta', 'π': 'pi',
    'ν': 'nu', 'γ': 'gamma', 'κ': 'kappa', 'ε': 'epsilon', 'β': 'beta',
    'ρ': 'rho', 'ω': 'omega', 'χ': 'chi'
}

# IDs for special tokens for encoding inputs of the decoder models
EXTRA_ID_0 = '<extra_id_0>'
EXTRA_ID_1 = '<extra_id_1>'
