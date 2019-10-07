# Contributions are welcome!

1) Please take a look at the LICENSE (it's Apache 2.0)

2) Make sure you sign your commits. E.g. use ``git commit -s`` when commiting

3) Make sure all unittests finish successfully before sending PR

4) Send your Pull Request to `master` branch


# Collection Guidelines
Collection is a logical grouping of related Neural Modules. It is a grouping of modules that share a domain area or semantics. At the basic level, collection is a python package installable via pip.
When contributing module to a collection, please make sure it belongs to that category. If you would like to start a new one and contribute back to the platform, you are very welcome to do so. Collection package should be named “nemo_<collection-name>”. Collections can depend on other collections and have new types defined. Neural Types for input and output need to be clearly defined in documentation. 

Please note that CI needs to pass for all the modules and collections.

# Citation
If you are using NeMo please cite the following publication

@misc{nemo2019,
    title={NeMo: a toolkit for building AI applications using Neural Modules},
    author={Oleksii Kuchaiev and Jason Li and Huyen Nguyen and Oleksii Hrinchuk and Ryan Leary and Boris Ginsburg and Samuel Kriman and Stanislav Beliaev and Vitaly Lavrukhin and Jack Cook and Patrice Castonguay and Mariya Popova and Jocelyn Huang and Jonathan M. Cohen},
    year={2019},
    eprint={1909.09577},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}