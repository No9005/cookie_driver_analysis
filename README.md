# cookie_driver_analysis
An analysis of the candy-power-ranking dataset (link to the original dataset in **data/README.md**).

## Objectives
The target was to extract the importance of candy characteristics by using a key driver analysis.

## Brief overview of the procedure
After the dataset was loaded and checked for integrity, a test-train split was performed.
The prerequisites for various regression algorithms were checked on this split.
The VIF score was also tested to assess the multicollinearity of the variables (to deal with it, a dominance analysis was performed).

Several learning algorithms were tested on the train data via cross-validation.
The best learner among those tested, which also showed the lowest overfitting tendency, was used as the basis for the dominance analysis.

Finally, a few additional analyses were performed.

# Installation & startup
A virutal environment was used to install the required python packages.
If you use **pipenv** on a **linux** machine, you can use the following installation process:

----------------------------
>> cd *PATH-TO-GIT-DIRECTORY* <br>
>> pipenv install <br>
>> pipenv run jupyter notebook py_files/analysis.ipynb
----------------------------

Please make sure to install the required python version (3.8) first.
During the installation, **pipenv** should show you the location of your virtual environment.
You can use a virtual environment manager of your choice if you do not like to use **pipenv**.

# Deinstalling
Uninstalling is as easy as deleting the virtual environment directory.
Every installed package is installed in that particular **pipenv** **venv** directory.

To uninstall the required packages with **pipenv** (on a **linux** machine):

--------------------------
>> cd *PATH-TO-GIT-DIRECTORY* <br>
>> pipenv --venv <br>
>> rm -r *VENV-PATH-SHOWN-BY-PIPENV**
---------------------------

As an alternative you can delete the directory (shown by pipenv --venv) manually by using your computers file browser.

# Alternative to installing the packages
If you do not want to install the required python packages, you can use the converted code file under *output/*.
This .html file contains all the code and the visual output from the jupyter notebook.
