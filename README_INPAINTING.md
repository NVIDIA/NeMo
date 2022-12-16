This is a walkthrough on how to get started on all things inpainting in this repo.

### Installation:

first install all python dependencies (into a virtualenv is recommended):

```bash
pip install Cython
pip install -r requirements.txt --no-deps
```

There are some native libraries that are needed. Sadly I didn't record which ones those are. Any reader following these steps are encouraged to list them here after getting everything set up


### Data Installation:
get the lJspeech data

```bash
python scripts/dataset_processing/tts/ljspeech/get_data.py --data-root data/
```


(optional) download cached computed data
download from [here](https://drive.google.com/file/d/1lyZHm2KdRoBJgW2XuCV9q_EaXquw9wgc/view?usp=sharing)

then unzip and place as `data/LJSpeech-1.1/data_cache` this will speed up the first epoch on the machine greatly


### Running training

```bash
python examples/tts/inpainting.py trainer.accelerator=gpu
```
have a look at the [hydra docs](https://hydra.cc/docs/intro/) to see how you can override some of the fields in the config yamls to quickly change hyperparameters

once the training is running, logs and checkpoints will be saved at `nemo_experiments/Inpainter/{date_time}`, you can use tensorboard to track the progress of the model.


### Developement and Debugging

If you want to debug or improve the training pipeline. run:

```bash
python examples/tts/inpainting.py -cn inpainting_tiny.yaml
```

which will use a really small model and tiny dataset to quickly check the code is running correctly.

You will have to create the smaller datasets which can be quickly done by running:
```bash
cat data/LJSpeech-1.1/test_manifest.json | head -n 32 > data/LJSpeech-1.1/test_manifest_tiny.json
cat data/LJSpeech-1.1/val_manifest.json | head -n 32 > data/LJSpeech-1.1/val_manifest_tiny.json
```
