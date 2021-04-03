# Roles

Pranay - Headed Python development, Report

Dylan - Headed Scala development, Report

Nathan - Helped with Python and Scala, Report


# Running the Code

We used Scala and Python for our project.

### Scala

The files will all be placed in the scalation modeling analytics package. ConcreteStuff, WineStuff, StabilityStuff and BikeStuff are all modeled after ExampleAutoMPG, and serve as imports for the datasets for the ML models. Each of the five files starting with "CSCI4360Project2" have test runs with every model inside of them, including the commands to run the individual tests.

### Python

The python file is an ipynb and so should be opened with either Jupyter Notebook or Google Colab. I would recommend using Google Colab since the required packages are already installed. 

The first cell contains a method at the end named runAllModels. This method should be customized to signify what models should be run for the dataset, the instructions on what to change are listed in the method. 

Each dataset we are using is set up in a separate cell after the first cell. Each of those cells calls the runAllModels method for that particular dataset, so just that cell needs to be run for models for that particular dataset to be built and all the related plots to be printed.
