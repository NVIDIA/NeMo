## Chinese NER sample data format

This directory contains sample data for Chinese Name Entity Recognition. Please refer to this data format and prepare your own data.

Note that this sample data should not be used to train the NER model.

The data should be splitted into 2 file: text.txt and labels.txt.

Each line in **text_train.txt** and **text_dev.txt** contains text sequences,
where words are separated with spaces.
**labels_train.txt** and **labels_dev.txt** Chinese BIO format labels for training and validation.

Each line of the files should follow the format: 
[WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt)
and [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt). 


