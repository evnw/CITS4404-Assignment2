# CITS4404-Assignment2

## Description of files:

### camo_worms_utils.py
Contains a number of helper functions and structure objects.
Mostly these were from Cara's original code but a lot of new methods have been added.

### CamoWorms_SSGA.ipynb

The most up to date algorithm version.
The algorithm stops after a certain amount of iterations.
Iterations and Population can be set at the top of the file.

Run all cells to get outputs.
5000 generations (iterations) and 250 population was used for most tests.
NOTE: Running can take approximately 6 minutes

Weightings for the cost function are editted at the top of the main loop block.


### CamoWorms_EA.ipynb and CamoWorms_GA.ipynb
Old versions of the Evolutionary Algorithm and Genetic Algorithm.
Highly outdated, and do not contain the most recent cost functions.
Kept for archival purposes only.

### How to run:
1) Download and unpack repo
2) Open the CamoWorms_SSGA.ipynb notebook.
3) Outputs are already present at the bottom of the file
4) Run all cells to reproduce outputs. NOTE: Takes approx. 5 mins
5) Change Iterations or Population at top of the file