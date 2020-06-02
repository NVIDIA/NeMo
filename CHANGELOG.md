# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--

============== Guiding Principles ==============

* Changelogs are for humans, not machines.
* There should be an entry for every single version.
* The same types of changes should be grouped.
* Versions and sections should be linkable.
* The latest version comes first.
* The release date of each version is displayed.
* Mention whether you follow Semantic Versioning.

============== Types of changes (keep the order) ==============

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.
* `Dependencies Update` in case of vulnerabilities.
* `Contributors` to thank the contributors that worked on this PR.

============== How To Update The Changelog for a New Release ==============

** Always Keep The Unreleased On Top **

To release a new version, please update the changelog as followed:
1. Rename the `Unreleased` Section to the Section Number
2. Recreate an `Unreleased` Section on top
3. Update the links at the very bottom

======================= START: TEMPLATE TO KEEP IN CASE OF NEED ===================

** DO NOT MODIFY THIS SECTION ! **

## [Unreleased]

### Added

### Changed

### Dependencies Update

### Deprecated

### Fixed

### Removed

### Security

### Contributors

** DO NOT MODIFY THIS SECTION ! **

======================= END: TEMPLATE TO KEEP IN CASE OF NEED ===================

-->

<!-- YOU CAN EDIT FROM HERE -->

## [Unreleased]

### Added
- Added NeMoModels class. Implemented in ASR collection: ASRConvCTCModel, and QuartzNet and JasperNet as its children - @okuchaiev
- Added multi-dataset data-layer and dataset.
([PR #538](https://github.com/NVIDIA/NeMo/pull/538)) - @yzhang123
- Online Data Augmentation for ASR Collection. ([PR #565](https://github.com/NVIDIA/NeMo/pull/565)) - @titu1994
- Speed augmentation on CPU, TimeStretch augmentation on CPU+GPU ([PR #594](https://github.com/NVIDIA/NeMo/pull/565)) - @titu1994
- Added TarredAudioToTextDataLayer, which allows for loading ASR datasets with tarred audio. Existing datasets can be converted with the `convert_to_tarred_audio_dataset.py` script. ([PR #602](https://github.com/NVIDIA/NeMo/pull/602))
- Online audio augmentation notebook in ASR examples ([PR #605](https://github.com/NVIDIA/NeMo/pull/605)) - @titu1994
- ContextNet Encoder + Decoder Initial Support ([PR #630](https://github.com/NVIDIA/NeMo/pull/630)) - @titu1994
- Added finetuning with Megatron-LM ([PR #601](https://github.com/NVIDIA/NeMo/pull/601)) - @ekmb
- Added documentation for 8 kHz model ([PR #632](https://github.com/NVIDIA/NeMo/pull/632)) - @jbalam-nv
- The Neural Graph is a high-level abstract concept empowering the users to build graphs consisting of many, interconnected Neural Modules. A user in his/her application can build any number of graphs, potentially spanning over the same modules. The import/export options combined with the lightweight API make Neural Graphs a perfect tool for rapid prototyping and experimentation. ([PR #413](https://github.com/NVIDIA/NeMo/pull/413)) - @tkornuta-nvidia
- Created the NeMo CV collection, added  MNIST and CIFAR10 thin datalayers, implemented/ported several general usage trainable and non-trainable modules, added several new ElementTypes ([PR #654](https://github.com/NVIDIA/NeMo/pull/654)) - @tkornuta-nvidia


### Changed
- quartznet and jasper ASR examples reworked into speech2text.py and speech2text_infer.py - @okuchaiev
- Syncs across workers at each step to check for NaN or inf loss. Terminates all workers if stop\_on\_nan\_loss is set (as before), lets Apex deal with it if apex.amp optimization level is O1 or higher, and skips the step across workers otherwise. ([PR #637](https://github.com/NVIDIA/NeMo/pull/637)) - @redoctopus
- Updated the callback system. Old callbacks will be deprecated in version 0.12. ([PR #615](https://github.com/NVIDIA/NeMo/pull/615)) - @blisc

### Dependencies Update

### Deprecated

### Fixed

### Removed

### Security

## [0.10.0] - 2020-04-03

### Added
- Roberta and Albert support added to GLUE script, data caching also added.
([PR #413](https://github.com/NVIDIA/NeMo/pull/413)) - @ekmb
- text classification notebook added
([PR #382](https://github.com/NVIDIA/NeMo/pull/382)) - @ericharper
- New Neural Type System documentation. Also added decorator to generate docs for input/output ports.
([PR #370](https://github.com/NVIDIA/NeMo/pull/370)) - @okuchaiev
- New Neural Type System and its tests.
([PR #307](https://github.com/NVIDIA/NeMo/pull/307)) - @okuchaiev
- Named tensors tuple module's output for graph construction.
([PR #268](https://github.com/NVIDIA/NeMo/pull/268)) - @stasbel
- Introduced the `deprecated` decorator.
([PR #298](https://github.com/NVIDIA/NeMo/pull/298)) - @tkornuta-nvidia
- Implemented new mechanisms for importing and exporting of module configuration (init_params) to configuration (yml)
files, along with unit tests, examples and tutorials
([PR #339](https://github.com/NVIDIA/NeMo/pull/339)) - @tkornuta-nvidia
- Speech Commands support.
([PR #375](https://github.com/NVIDIA/NeMo/pull/375)) - @titu1994

### Changed
- Refactoring of `nemo_nlp` collections:
([PR #368](https://github.com/NVIDIA/NeMo/pull/368)) - @VahidooX, @yzhang123, @ekmb
    - renaming and restructuring of files, folder, and functions in `nemo_nlp`
    - losses cleaned up. LossAggregatorNM moved to nemo/backends/pytorch/common/losses
 ([PR #316](https://github.com/NVIDIA/NeMo/pull/316)) - @VahidooX, @yzhang123, @ekmb
    - renaming and restructuring of files, folder, and functions in `nemo_nlp`
    - Updated licenses
- All collections changed to use New Neural Type System.
([PR #307](https://github.com/NVIDIA/NeMo/pull/307)) - @okuchaiev
- Additional Collections Repositories merged into core `nemo_toolkit` package.
([PR #289](https://github.com/NVIDIA/NeMo/pull/289)) - @DEKHTIARJonathan
- Refactor manifest files parsing and processing for re-using.
([PR #284](https://github.com/NVIDIA/NeMo/pull/284)) - @stasbel
- NeMo is not longer using pep8 code style rules. Code style rules are now enforced with `isort` and `black` incorporated into CI checks.
([PR #286](https://github.com/NVIDIA/NeMo/pull/286)) - @stasbel
- Major cleanup of Neural Module constructors (init), aiming at increasing the framework robustness: cleanup of NeuralModule initialization logic, refactor of trainer/actions (getting rid of local_params), fixes of several examples and unit tests, extraction and storing of intial parameters (init_params).
([PR #309](https://github.com/NVIDIA/NeMo/pull/309)) - @tkornuta-nvidia
- Updated nemo's use of the logging library. from nemo import logging is now the reccomended way of using the nemo logger. neural_factory.logger and all other instances of logger are now deprecated and planned for removal in the next version. Please see PR 267 for complete change information.
([PR #267](https://github.com/NVIDIA/NeMo/pull/267), [PR #283](https://github.com/NVIDIA/NeMo/pull/283), [PR #305](https://github.com/NVIDIA/NeMo/pull/305), [PR #311](https://github.com/NVIDIA/NeMo/pull/311)) - @blisc
- Changed Distributed Data Parallel from Apex to Torch
([PR #336](https://github.com/NVIDIA/NeMo/pull/336)) - @blisc

- Added TRADE (dialogue state tracking model) on MultiWOZ dataset
([PR #322](https://github.com/NVIDIA/NeMo/pull/322)) - @chiphuyen, @VahidooX
- Question answering:
([PR #390](https://github.com/NVIDIA/NeMo/pull/390)) - @yzhang123
    - Changed question answering task to use Roberta and Albert as alternative backends to Bert
    - Added inference mode that does not require ground truth labels

### Dependencies Update
- Added dependency on `wrapt` (the new version of the `deprecated` warning) - @tkornuta-nvidia, @DEKHTIARJonathan

### Deprecated

### Fixed
- Critical fix of the training action on CPU
([PR #308](https://github.com/NVIDIA/NeMo/pull/309)) - @tkornuta-nvidia
- Fixed issue in Tacotron 2 prenet
([PR #444](https://github.com/NVIDIA/NeMo/pull/444)) - @blisc

### Removed
- gradient_predivide_factor arg of train() now has no effect
([PR #336](https://github.com/NVIDIA/NeMo/pull/336)) - @blisc
- Dropped support of the following ASR configs: jasper10x4.yaml, quartznet10x5.yaml, quartznet15x5_in.yaml, quartznet5x3.yaml, quartznet5x5.yaml, quartznet_an4.yaml. They are moved to experimental/configs and can still be used with v0.9 for use in replicating paper results
([PR #354](https://github.com/NVIDIA/NeMo/pull/354)) - @blisc

### Security

### Contributors

## [0.9.0] - 2019-12-16

This release contains new features, new models and quality improvements for NeMo.

### Highlights

* Added "nemo_tts" - a Speech Synthesis collection with necessary modules for Tacotron2 and WaveGlow
* Added Mandarin support into nemo_asr and nemo_nlp
* Updated ASR and TTS checkpoints including Mandarin ASR
* Documentation now translated to Mandarin https://nvidia.github.io/NeMo/chinese/intro.html
* Export functionality for deployment
* General improvements and bug-fixes

## [0.8.2] - 2019-11-14

This is a quality improvement release for NeMo.

### Highlights

* Bugfixes
* Support for Pytorch 1.3

## [0.8.1] - 2019-12-16

This is a quality improvement release for NeMo.

### Highlights

* Added introductory ASR tutorial explaining how to get started with deep learning for ASR
* Re-organization of NeMo NLP library
* More efficient BERT pre-training implementation
* General improvements and bugfixes
* Support for CPU-only scenario

### Special thanks to our external contributors
 - David Pollack @dhpollack
 - Harisankar Haridas @harisankarh
 - Dilshod Tadjibaev @antimora

## [0.8.0] - 2019-12-16

The first public release of NVIDIA Neural Modules: NeMo.

This release also includes nemo_asr'' and nemo_nlp'' collections for Speech Recognition and Natural Language Processing.

Please refer to the documentation here: https://nvidia.github.io/NeMo/

[Unreleased]: https://github.com/NVIDIA/NeMo/compare/v0.10.0...master
[0.10.0]: https://github.com/NVIDIA/NeMo/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/NVIDIA/NeMo/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/NVIDIA/NeMo/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/NVIDIA/NeMo/compare/r0.8...v0.8.1
[0.8.0]: https://github.com/NVIDIA/NeMo/tree/r0.8
