#!/bin/bash
 #Train scp

lda_dim=200
DIR=$1
stage=$2
train_scp=$DIR/train.scp
dev_scp=$DIR/dev.scp

trail_file=$DIR/trials_1m
cd $KALDI_ROOT/egs/voxceleb/v2
. path.sh
. cmd.sh
cd -

if [ $stage -le 1 ]; then
    ivector-mean scp:$train_scp $DIR/mean.vec 

    echo "Training LDA"
    $train_cmd $DIR/log ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_scp ark:- |" \
    ark:$DIR/utt2spk $DIR/transform.mat || exit 1;

    echo "TRAINING PLDA"
    $train_cmd $DIR/log ivector-compute-plda ark:$DIR/spk2utt \
    "ark:ivector-subtract-global-mean scp:$train_scp ark:- | transform-vec $DIR/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" $DIR/plda || exit 1;
fi

if [ $stage -le 2 ]; then
    echo "SCORING"
    sed 's/{}/average/' $trail_file > $DIR/temp_trail
    $train_cmd $DIR/log ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $DIR/plda - |" \
    "ark:ivector-subtract-global-mean $DIR/mean.vec scp:$dev_scp ark:- | transform-vec $DIR/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $DIR/mean.vec scp:$dev_scp ark:- | transform-vec $DIR/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$DIR/temp_trail' | cut -d\  --fields=1,2 |" $DIR/scores || exit 1;

    paste -d' ' <(awk '{print $3}' $DIR/scores) <(awk '{print $3}' $trail_file) > $DIR/final_score

    eer=`compute-eer <($KALDI_ROOT/egs/voxceleb/v2/local/prepare_for_eer.py  $DIR/temp_trail $DIR/scores) 2> /dev/null` 
    # eer=`compute-eer $DIR/final_score 2> /dev/null`
    mindcf=`$KALDI_ROOT/egs/voxceleb/v2/sid/compute_min_dcf.py $DIR/scores $DIR/temp_trail 2> /dev/null`
    echo "EER: $eer%"
    echo "minDCF: $mindcf"

fi