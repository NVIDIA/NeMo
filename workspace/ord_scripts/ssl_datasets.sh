# Data paths
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
SSL_ML_DATA=${DATA_DIR}/SSL/multilingual

############# Youtube Data #############
## German
YT_DE_ROOT=${SSL_ML_DATA}/yt_100k/DE/train_tarred
YT_TRAIN_DE_MANIFEST="[[${YT_DE_ROOT}/bucket1/tarred_audio_manifest.json],[${YT_DE_ROOT}/bucket2/tarred_audio_manifest.json],[${YT_DE_ROOT}/bucket3/tarred_audio_manifest.json],[${YT_DE_ROOT}/bucket4/tarred_audio_manifest.json]]"
YT_TRAIN_DE_FILEPATH="[[${YT_DE_ROOT}/bucket1/audio__OP_0..511_CL_.tar],[${YT_DE_ROOT}/bucket2/audio__OP_0..511_CL_.tar],[${YT_DE_ROOT}/bucket3/audio__OP_0..511_CL_.tar],[${YT_DE_ROOT}/bucket4/audio__OP_0..511_CL_.tar]]"

## Spanish
YT_ES_ROOT=${SSL_ML_DATA}/yt_100k/ES/tarred_train
YT_TRAIN_ES_MANIFEST="[[${YT_ES_ROOT}/bucket1/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket2/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket3/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket4/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket5/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket6/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket7/tarred_audio_manifest.json],[${YT_ES_ROOT}/bucket8/tarred_audio_manifest.json]]"
YT_TRAIN_ES_FILEPATH="[[${YT_ES_ROOT}/bucket1/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket2/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket3/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket4/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket5/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket6/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket7/audio__OP_0..511_CL_.tar],[${YT_ES_ROOT}/bucket8/audio__OP_0..511_CL_.tar]]"

## French
YT_FR_ROOT=${SSL_ML_DATA}/yt_100k/FR/tarred_train
YT_TRAIN_FR_MANIFEST="[[${YT_FR_ROOT}/bucket1/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket2/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket3/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket4/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket5/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket6/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket7/tarred_audio_manifest.json],[${YT_FR_ROOT}/bucket8/tarred_audio_manifest.json]]"
YT_TRAIN_FR_FILEPATH="[[${YT_FR_ROOT}/bucket1/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket2/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket3/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket4/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket5/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket6/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket7/audio__OP_0..511_CL_.tar],[${YT_FR_ROOT}/bucket8/audio__OP_0..511_CL_.tar]]"


############## Librilight-EN #############
LL_TRAIN_EN_MANIFEST=$DATA_DIR/ASR/unlab-60k_seg_tarred/tarred_audio_manifest.json
LL_TRAIN_EN_FILEPATH=$DATA_DIR/ASR/unlab-60k_seg_tarred/audio__OP_0..2047_CL_.tar


############# Voxlingual-107 #############
VOXLING_ROOT=$SSL_ML_DATA/voxlingua107
VOXLING_TRAIN_V107_MANIFEST=$VOXLING_ROOT/train_tarred/tarred_audio_manifest.json
VOXLING_TRAIN_V107_FILEPATH=$VOXLING_ROOT/train_tarred/audio__OP_0..1023_CL_.tar

VOXLING_VAL_V107_MANIFEST=$VOXLING_ROOT/processed_dev/dev_manifest.json


############## Voxpopuli #############
VOXPOP_ROOT=$SSL_ML_DATA/voxpopuli

## English
VOXPOP_TRAIN_EN_MANIFEST=$VOXPOP_ROOT/en_tarred/tarred_audio_manifest.json
VOXPOP_TRAIN_EN_FILEPATH=$VOXPOP_ROOT/en_tarred/audio__OP_0..8191_CL_.tar

## German
VOXPOP_TRAIN_DE_MANIFEST=$VOXPOP_ROOT/de_tarred/tarred_audio_manifest.json
VOXPOP_TRAIN_DE_FILEPATH=$VOXPOP_ROOT/de_tarred/audio__OP_0..8191_CL_.tar

## Italian
VOXPOP_TRAIN_IT_MANIFEST=$VOXPOP_ROOT/it_tarred/tarred_audio_manifest.json
VOXPOP_TRAIN_IT_FILEPATH=$VOXPOP_ROOT/it_tarred/audio__OP_0..8191_CL_.tar

## Polish
VOXPOP_TRAIN_PL_MANIFEST=$VOXPOP_ROOT/pl_tarred/tarred_audio_manifest.json
VOXPOP_TRAIN_PL_FILEPATH=$VOXPOP_ROOT/pl_tarred/audio__OP_0..8191_CL_.tar

## Portuguese
VOXPOP_TRAIN_PT_MANIFEST=$VOXPOP_ROOT/pt_tarred/tarred_audio_manifest.json
VOXPOP_TRAIN_PT_FILEPATH=$VOXPOP_ROOT/pt_tarred/audio__OP_0..8191_CL_.tar

## French
VOXPOP_TRAIN_FR_MANIFEST=$VOXPOP_ROOT/fr/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_FR_FILEPATH=$VOXPOP_ROOT/fr/tarred_train/audio__OP_0..8191_CL_.tar

# Cs (Czech)
# VOXPOP_TRAIN_CS_MANIFEST=$VOXPOP_ROOT/cs/tarred_train/tarred_audio_manifest.json
# VOXPOP_TRAIN_CS_FILEPATH=$VOXPOP_ROOT/cs/tarred_train/audio__OP_0..8191_CL_.tar

## Finnish (fi)
VOXPOP_TRAIN_FI_MANIFEST=$VOXPOP_ROOT/fi/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_FI_FILEPATH=$VOXPOP_ROOT/fi/tarred_train/audio__OP_0..8191_CL_.tar

## Hr (Croatian)
VOXPOP_TRAIN_HR_MANIFEST=$VOXPOP_ROOT/hr/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_HR_FILEPATH=$VOXPOP_ROOT/hr/tarred_train/audio__OP_0..8191_CL_.tar

## Sk (Slovak)
VOXPOP_TRAIN_SK_MANIFEST=$VOXPOP_ROOT/sk/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_SK_FILEPATH=$VOXPOP_ROOT/sk/tarred_train/audio__OP_0..8191_CL_.tar

## Slovene (sl)
VOXPOP_TRAIN_SL_MANIFEST=$VOXPOP_ROOT/sl/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_SL_FILEPATH=$VOXPOP_ROOT/sl/tarred_train/audio__OP_0..8191_CL_.tar

## Et (Estonian)
VOXPOP_TRAIN_ET_MANIFEST=$VOXPOP_ROOT/et/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_ET_FILEPATH=$VOXPOP_ROOT/et/tarred_train/audio__OP_0..8191_CL_.tar

## Bg (Bulgarian)
VOXPOP_TRAIN_BG_MANIFEST=$VOXPOP_ROOT/bg/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_BG_FILEPATH=$VOXPOP_ROOT/bg/tarred_train/audio__OP_0..8191_CL_.tar

## Greek (el)
VOXPOP_TRAIN_EL_MANIFEST=$VOXPOP_ROOT/el/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_EL_FILEPATH=$VOXPOP_ROOT/el/tarred_train/audio__OP_0..8191_CL_.tar

## Lv (Latvian)
VOXPOP_TRAIN_LV_MANIFEST=$VOXPOP_ROOT/lv/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_LV_FILEPATH=$VOXPOP_ROOT/lv/tarred_train/audio__OP_0..8191_CL_.tar

## Maltese (mt)
VOXPOP_TRAIN_MT_MANIFEST=$VOXPOP_ROOT/mt/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_MT_FILEPATH=$VOXPOP_ROOT/mt/tarred_train/audio__OP_0..8191_CL_.tar

## Swedish (sv)
VOXPOP_TRAIN_SV_MANIFEST=$VOXPOP_ROOT/sv/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_SV_FILEPATH=$VOXPOP_ROOT/sv/tarred_train/audio__OP_0..8191_CL_.tar

## Da (Danish)
VOXPOP_TRAIN_DA_MANIFEST=$VOXPOP_ROOT/da/tarred_train/tarred_audio_manifest.json
VOXPOP_TRAIN_DA_FILEPATH=$VOXPOP_ROOT/da/tarred_train/audio__OP_0..8191_CL_.tar

############# Babel #############
BABEL_ROOT=$SSL_ML_DATA/babel

## Turkish
BABEL_TRAIN_TR_MANIFEST=$BABEL_ROOT/tr/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_TR_FILEPATH=$BABEL_ROOT/tr/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_TR_MANIFEST=$BABEL_ROOT/tr/dev_manifest_conv_0.json

## Tagalog
BABEL_TRAIN_TL_MANIFEST=$BABEL_ROOT/tl/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_TL_FILEPATH=$BABEL_ROOT/tl/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_TL_MANIFEST=$BABEL_ROOT/tl/dev_manifest_conv_0.json

## Cebuano
BABEL_TRAIN_CEB_MANIFEST=$BABEL_ROOT/ceb/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_CEB_FILEPATH=$BABEL_ROOT/ceb/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_CEB_MANIFEST=$BABEL_ROOT/ceb/dev_manifest_conv_0.json

## Kazakh
BABEL_TRAIN_KK_MANIFEST=$BABEL_ROOT/kk/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_KK_FILEPATH=$BABEL_ROOT/kk/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_KK_MANIFEST=$BABEL_ROOT/kk/dev_manifest_conv_0.json

## Guarani
BABEL_TRAIN_GN_MANIFEST=$BABEL_ROOT/gn/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_GN_FILEPATH=$BABEL_ROOT/gn/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_GN_MANIFEST=$BABEL_ROOT/gn/dev_manifest_conv_0.json

## Ig (Igbo)
BABEL_TRAIN_IG_MANIFEST=$BABEL_ROOT/ig/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_IG_FILEPATH=$BABEL_ROOT/ig/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_IG_MANIFEST=$BABEL_ROOT/ig/dev_manifest_conv_0.json

## Zu (Zulu)
BABEL_TRAIN_ZU_MANIFEST=$BABEL_ROOT/zu/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_ZU_FILEPATH=$BABEL_ROOT/zu/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_ZU_MANIFEST=$BABEL_ROOT/zu/dev_manifest_conv_0.json

## Assamese
BABEL_TRAIN_AS_MANIFEST=$BABEL_ROOT/as/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_AS_FILEPATH=$BABEL_ROOT/as/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_AS_MANIFEST=$BABEL_ROOT/as/dev_manifest_conv_0.json

## Georgian
BABEL_TRAIN_KA_MANIFEST=$BABEL_ROOT/ka/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_KA_FILEPATH=$BABEL_ROOT/ka/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_KA_MANIFEST=$BABEL_ROOT/ka/dev_manifest_conv_0.json

# Bengali
BABEL_TRAIN_BN_MANIFEST=$BABEL_ROOT/bn/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_BN_FILEPATH=$BABEL_ROOT/bn/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_BN_MANIFEST=$BABEL_ROOT/bn/dev_manifest_conv_0.json

# Pashto
BABEL_TRAIN_PS_MANIFEST=$BABEL_ROOT/ps/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_PS_FILEPATH=$BABEL_ROOT/ps/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_PS_MANIFEST=$BABEL_ROOT/ps/dev_manifest_conv_0.json

# Lithuanian
BABEL_TRAIN_LT_MANIFEST=$BABEL_ROOT/lt/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_LT_FILEPATH=$BABEL_ROOT/lt/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_LT_MANIFEST=$BABEL_ROOT/lt/dev_manifest_conv_0.json

# Tamil
BABEL_TRAIN_TA_MANIFEST=$BABEL_ROOT/ta/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_TA_FILEPATH=$BABEL_ROOT/ta/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_TA_MANIFEST=$BABEL_ROOT/ta/dev_manifest_conv_0.json

# Telugu
BABEL_TRAIN_TE_MANIFEST=$BABEL_ROOT/te/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_TE_FILEPATH=$BABEL_ROOT/te/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_TE_MANIFEST=$BABEL_ROOT/te/dev_manifest_conv_0.json

# Tok Pisin
BABEL_TRAIN_TPI_MANIFEST=$BABEL_ROOT/tpi/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_TPI_FILEPATH=$BABEL_ROOT/tpi/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_TPI_MANIFEST=$BABEL_ROOT/tpi/dev_manifest_conv_0.json

# Kurmanji Kurdish
BABEL_TRAIN_KU_MANIFEST=$BABEL_ROOT/ku/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_KU_FILEPATH=$BABEL_ROOT/ku/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_KU_MANIFEST=$BABEL_ROOT/ku/dev_manifest_conv_0.json

# Amharic
BABEL_TRAIN_AM_MANIFEST=$BABEL_ROOT/am/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_AM_FILEPATH=$BABEL_ROOT/am/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_AM_MANIFEST=$BABEL_ROOT/am/dev_manifest_conv_0.json

# Lao
BABEL_TRAIN_LO_MANIFEST=$BABEL_ROOT/lo/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_LO_FILEPATH=$BABEL_ROOT/lo/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_LO_MANIFEST=$BABEL_ROOT/lo/dev_manifest_conv_0.json

# Vietnamese
BABEL_TRAIN_VI_MANIFEST=$BABEL_ROOT/vi/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_VI_FILEPATH=$BABEL_ROOT/vi/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_VI_MANIFEST=$BABEL_ROOT/vi/dev_conv_0.json

## Swahili
BABEL_TRAIN_SW_MANIFEST=$BABEL_ROOT/sw/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_SW_FILEPATH=$BABEL_ROOT/sw/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_SW_MANIFEST=$BABEL_ROOT/sw/dev_manifest_conv_0.json

# Cantonese
BABEL_TRAIN_YUE_MANIFEST=$BABEL_ROOT/yue/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_YUE_FILEPATH=$BABEL_ROOT/yue/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_YUE_MANIFEST=$BABEL_ROOT/yue/dev_manifest_conv_0.json

# Haitian Creole
BABEL_TRAIN_HT_MANIFEST=$BABEL_ROOT/ht/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_HT_FILEPATH=$BABEL_ROOT/ht/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_HT_MANIFEST=$BABEL_ROOT/ht/dev_manifest_conv_0.json

# Mongolian
BABEL_TRAIN_MN_MANIFEST=$BABEL_ROOT/mn/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_MN_FILEPATH=$BABEL_ROOT/mn/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_MN_MANIFEST=$BABEL_ROOT/mn/dev_manifest_conv_0.json

# Javanese
BABEL_TRAIN_JW_MANIFEST=$BABEL_ROOT/jw/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_JW_FILEPATH=$BABEL_ROOT/jw/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_JW_MANIFEST=$BABEL_ROOT/jw/dev_manifest_conv_0.json

# Dholuo
BABEL_TRAIN_LUO_MANIFEST=$BABEL_ROOT/luo/tarred_train/tarred_audio_manifest.json
BABEL_TRAIN_LUO_FILEPATH=$BABEL_ROOT/luo/tarred_train/audio__OP_0..511_CL_.tar
BABEL_VAL_LUO_MANIFEST=$BABEL_ROOT/luo/dev_manifest_conv_0.json


############# MLS #############
MLS_ROOT=$SSL_ML_DATA/mls

## English
MLS_TRAIN_EN_MANIFEST=$MLS_ROOT/en/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_EN_FILEPATH=$MLS_ROOT/en/tarred_train/audio__OP_0..511_CL_.tar

## German
MLS_TRAIN_DE_MANIFEST=$MLS_ROOT/de/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_DE_FILEPATH=$MLS_ROOT/de/tarred_train/audio__OP_0..511_CL_.tar

## Spanish
MLS_TRAIN_ES_MANIFEST=$MLS_ROOT/es/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_ES_FILEPATH=$MLS_ROOT/es/tarred_train/audio__OP_0..511_CL_.tar

## French
MLS_TRAIN_FR_MANIFEST=$MLS_ROOT/fr/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_FR_FILEPATH=$MLS_ROOT/fr/tarred_train/audio__OP_0..511_CL_.tar

## Italian
MLS_TRAIN_IT_MANIFEST=$MLS_ROOT/it/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_IT_FILEPATH=$MLS_ROOT/it/tarred_train/audio__OP_0..511_CL_.tar

## Polish
MLS_TRAIN_PL_MANIFEST=$MLS_ROOT/pl/tarred_train/tarred_audio_manifest.json
MLS_TRAIN_PL_FILEPATH=$MLS_ROOT/pl/tarred_train/audio__OP_0..511_CL_.tar


############# MMLPC #############
MMLPC_ROOT=$DATA/ASR/MMLPC

## English
# MMLPC_TRAIN_EN_MANIFEST=$MMLPC_ROOT/en/tarred_train/tarred_audio_manifest.json
# MMLPC_TRAIN_EN_FILEPATH=$MMLPC_ROOT/en/tarred_train/audio__OP_0..511_CL_.tar
MMLPC_VAL_EN_MANIFEST=$MMLPC_ROOT/en/val_test/val_en_pcstrip.json

## German
# MMLPC_TRAIN_DE_MANIFEST=$MMLPC_ROOT/de/tarred_train/tarred_audio_manifest.json
# MMLPC_TRAIN_DE_FILEPATH=$MMLPC_ROOT/de/tarred_train/audio__OP_0..511_CL_.tar
MMLPC_VAL_DE_MANIFEST=$MMLPC_ROOT/de/dev_de_pcstrip.json

## Spanish
# MMLPC_TRAIN_ES_MANIFEST=$MMLPC_ROOT/es/nemo_sp_asr_set_3pt0/tarred_train/tarred_audio_manifest.json
# MMLPC_TRAIN_ES_FILEPATH=$MMLPC_ROOT/es/nemo_sp_asr_set_3pt0/tarred_train/audio__OP_0..511_CL_.tar
MMLPC_VAL_ES_MANIFEST=$MMLPC_ROOT/es/nemo_sp_asr_set_3pt0/dev/val_es_pcstrip.json

## French
# MMLPC_TRAIN_FR_MANIFEST=$MMLPC_ROOT/fr/train_tarred/tarred_audio_manifest.json
# MMLPC_TRAIN_FR_FILEPATH=$MMLPC_ROOT/fr/train_tarred/audio__OP_0..255_CL_.tar
MMLPC_VAL_FR_MANIFEST=$MMLPC_ROOT/fr/devtest/val_fr_pcstrip.json

## Japanese
MMLPC_TRAIN_JA_MANIFEST=$MMLPC_ROOT/ja/tarred_train/tarred_audio_manifest.json
MMLPC_TRAIN_JA_FILEPATH=$MMLPC_ROOT/ja/tarred_train/audio__OP_0..4095_CL_.tar
MMLPC_VAL_JA_MANIFEST=$MMLPC_ROOT/ja/devtest/val_ja_pcstrip.json

## Italian
# MMLPC_TRAIN_IT_MANIFEST=$MMLPC_ROOT/it/tarred_train/tarred_audio_manifest.json
# MMLPC_TRAIN_IT_FILEPATH=$MMLPC_ROOT/it/tarred_train/audio__OP_0..511_CL_.tar
MMLPC_VAL_IT_MANIFEST=$MMLPC_ROOT/it/dev/val_it_pcstrip.json

## Russian
# MMLPC_TRAIN_RU_MANIFEST=$MMLPC_ROOT/ru/v2/tarred_train/tarred_audio_manifest.json
# MMLPC_TRAIN_RU_FILEPATH=$MMLPC_ROOT/ru/v2/tarred_train/audio__OP_0..511_CL_.tar
# MMLPC_VAL_RU_MANIFEST=$MMLPC_ROOT/ru/v2/dev/val_ru_pcstrip.json
 

############# MCV12 #############
MCV12_ROOT=$SSL_ML_DATA/mcv12

## English
MCV12_TRAIN_EN_MANIFEST=$MCV12_ROOT/en/tarred_train/tarred_audio_manifest.json
MCV12_TRAIN_EN_FILEPATH=$MCV12_ROOT/en/tarred_train/audio__OP_0..511_CL_.tar

## German
MCV12_TRAIN_DE_MANIFEST=$MCV12_ROOT/de/tarred_train/tarred_audio_manifest.json
MCV12_TRAIN_DE_FILEPATH=$MCV12_ROOT/de/tarred_train/audio__OP_0..511_CL_.tar

## Spanish
MCV12_TRAIN_ES_MANIFEST=$MCV12_ROOT/es/tarred_train/tarred_audio_manifest.json
MCV12_TRAIN_ES_FILEPATH=$MCV12_ROOT/es/tarred_train/audio__OP_0..511_CL_.tar

## French
MCV12_TRAIN_FR_MANIFEST=$MCV12_ROOT/fr/tarred_train/tarred_audio_manifest.json
MCV12_TRAIN_FR_FILEPATH=$MCV12_ROOT/fr/tarred_train/audio__OP_0..511_CL_.tar


############# SUNO #############
SUNO_ROOT=$DATA/multi_speaker

SUNO_20K_TRAIN_EN_MANIFEST=$SUNO_ROOT/suno_20k/train_tarred/tarred_audio_manifest.json
SUNO_20K_TRAIN_EN_FILEPATH=$SUNO_ROOT/suno_20k/train_tarred/audio__OP_0..511_CL_.tar
SUNO_20K_VAL_EN_MANIFEST=$SUNO_ROOT/suno_20k/dev_manifest.json

SUNO_100K_TRAIN_EN_MANIFEST=$SUNO_ROOT/suno_100k/train_tarred/tarred_audio_manifest.json
SUNO_100K_TRAIN_EN_FILEPATH=$SUNO_ROOT/suno_100k/train_tarred/audio__OP_0..511_CL_.tar
SUNO_100K_VAL_EN_MANIFEST=$SUNO_ROOT/suno_100k/dev_manifest.json




