ASR Tutorial Notebooks
----------------------

Table of Contents
-----------------
1. [Introduction to End-To-End Automatic Speech Recognition](./1_ASR_tutorial_using_NeMo.ipynb)

The introduction covers the basics of end-to-end automatic speech recognition, and how to get started with ASR using NeMo.
We recommend that you start with that if you are either new to ASR, or new to NeMo.

You should be able to import the notebook from Google Colab by using the "Upload from GitHub" option.

2. [Online Automatic Speech Recognition from a Microphone](./2_Online_ASR_Microphone_Demo.ipynb)

The notebook demonstrates automatic speech recognition (ASR) from a microphone's stream in NeMo.

It is **not a recommended** way to do inference in production workflows. If you are interested in a production-level inference using NeMo ASR models, please sign-up to [Jarvis early access program](https://developer.nvidia.com/nvidia-jarvis)

3. [Speech Commands in NeMo](./3_Speech_Commands_using_NeMo.ipynb)

This tutorial builds upon the introduction to ASR and covers the basics of speech command detection using the Google Speech Commands dataset.
Inspite of the small size of the model, advanced augmentation schemes such as SpecAugment can deliver high performance models.

We further analyse classification errors made by the model, and listen to samples which are predicted with least confidence by the trained model.
This exercise can be valuable when attempting to diagnose issues with the model or inspecting the dataset for inaccurate labelling. 
