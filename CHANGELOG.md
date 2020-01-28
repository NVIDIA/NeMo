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
- Named tensors tuple module's output for graph construction.
([PR #268](https://github.com/NVIDIA/NeMo/pull/268)) - @stasbel
- Convenient commands for code style testing and auto-reformatting with `isort` and `black`. 
([PR #284](https://github.com/NVIDIA/NeMo/pull/284)) - @stasbel

### Changed
- Additional Collections Repositories merged into core `nemo_toolkit` package.
([PR #289](https://github.com/NVIDIA/NeMo/pull/289)) - @DEKHTIARJonathan
- Refactor manifest files parsing and processing for re-using.
([PR #284](https://github.com/NVIDIA/NeMo/pull/284)) - @stasbel

### Dependencies Update

### Deprecated

### Fixed

### Removed

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

[Unreleased]: https://github.com/NVIDIA/NeMo/compare/v0.9.0...master
[0.9.0]: https://github.com/NVIDIA/NeMo/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/NVIDIA/NeMo/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/NVIDIA/NeMo/compare/r0.8...v0.8.1
[0.8.0]: https://github.com/NVIDIA/NeMo/tree/r0.8