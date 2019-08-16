# Table of contents

- [Getting Started](#getting-started)
- [Template Details](#template-details)
    - [Folder Structure](#folder-structure)
- [Acknowledgements](#acknowledgements)

# Getting Started
This template allows you to simply build and train deep learning models with checkpoints and tensorboard visualization.

In order to use the template you have to:
1. Define a generator class.
2. Define a model class that inherits from BaseModel.
3. Define a trainer class that inherits.
4. Define a configuration file with the parameters needed in an experiment.
5. Run the model using:
```shell
python main.py -c [path to configuration file]
```

# Template Details

## Folder Structure

```
├── main.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_evaluater.py   - this file contains the abstract class of the evaluator.
│   ├── base_generator.py   - this file contains the abstract class of the generator.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_trainer.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── trainer.py
│
|
├── generators         - this folder contains the data loaders of your project.
│   └── generator.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── config.json
│
│
├── data            - this folder might contain the data of your project.
│
│
└── utils               - this folder contains any utils you need.
     └── ...

```

# Acknowledgements
This project template is based on [MrGemy95](https://github.com/MrGemy95)'s 
[Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template)
and [Ahmkel](https://github.com/Ahmkel)'s [Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)


Thanks for my colleagues [Mahmoud Khaled](https://github.com/MahmoudKhaledAli), 
[Ahmed Waleed](https://github.com/Rombux) and 
[Ahmed El-Gammal](https://github.com/AGammal) and
[Ahmkel](https://github.com/Ahmkel)
who worked on the initial project that spawned this template.
