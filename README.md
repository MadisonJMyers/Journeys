# Setting up and Running SystemML

## Why SystemML?

### During my time as a UC Berkeley grad student, we were taught R and Python, so that is what I'm familar with. SystemML runs with R and Python. Being new to computer science and wanting to jump straight into the data doesn't allow me much time to hack into Spark and figure out how to write high-level math with big data. On SystemML you can write the math no matter how big the data is! Because I can access algorithms from files, it's easier to go from formulas and Python code to big data problems.
## Now let's get to my first dive into SystemML where I’ll focus on: overcoming assumptions.

### While I may still be very new to the tech world and all of its wonderful tutorials, an issue that I have consistently noticed thus far, is the long list of assumptions made in any step by step guide, particularly in setting up your environment. Many developers, data scientists and researchers are so advanced, they have forgotten what it’s like to be new! When writing tutorials, they assume that everything is set up and ready to go, but that’s not always the case. No need to worry with SystemML: I am here to help. Below is my very own step by step guide to running Apache SystemML on Jupyter notebook (with little to no assumptions).



# SystemML Jupyter Tutorial

### *If you are just starting out please read the following “setting up your environment” step. If you aren’t just starting out please skip to “run SystemML”, but make sure to install SystemML first!
## Setting up your environment.

### If you’re on a mac, you’ll want to install homebrew (http://brew.sh) if you haven’t already.
### Copy and paste the following into your terminal.

### OS X:
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
### Linux
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"  

### Now install Java (need Java 8).

brew tap caskroom/cask  
brew install Caskroom/cask/java  

### In order to install something on homebrew all you need to do is type "brew install" followed by what you want to install. See below.
### Follow-up by installing everything else you need.
### Install Spark.

brew tap homebrew/versions  
brew install apache-spark16  

### Install python 2 or 3.

### Install Python 2 with Jupyter, Matplotlib and Numpy
brew install python  
pip install jupyter matplotlib numpy  

### Install Python 3 with Jupyter, Matplotlib and Numpy
brew install python3  
pip3 install jupyter matplotlib numpy  

### Download SystemML.

### Go to the Apache SystemML downloads page and download the zip file (second file).

### This next step is optional, but it will make your life a lot easier.
### Set SYSTEMML_ HOME on your bash profile.

### First, use vim to create/edit your bash profile. Not sure what vim is? Check https://www.linux.com/learn/vim-101-beginners-guide-vim.

### We are going to insert our file where Spark and SystemML is stored into our bash profile. This will make it easier to access. First type:

vim .bash_profile  

### Now you are in your vim. First, type “i” for insert.

i  

### Now insert SystemML. Note: /Documents is where I saved my SystemML. Be sure that your file path is accurate.

export SYSTEMML_HOME=/Users/stc/Documents/systemml-0.10.0-incubating  

### Now type :wq to write the file and quit

:wq

### Make sure to open a new tab in terminal so that you make sure the changes have been made.
## Congrats! You’ve made it to the step where we run SystemML!


# Run SystemML flawlessly.


### In your browser, if you go to http://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide.html#jupyter-pyspark-notebook-example---poisson-nonnegative-matrix-factorization you will see a long line of code under “Nonnegative Matrix Factorization”.


### Take a look at this page if you want to understand the code more, but we only need to use part of it. In your terminal, type:

PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook" pyspark --master local[*] --driver-class-path $SYSTEMML_HOME/target/SystemML.jar --jars $SYSTEMML_HOME/target/SystemML.jar --conf "spark.driver.memory=12g" --conf spark.driver.maxResultSize=0 --conf spark.akka.frameSize=128 --conf spark.default.parallelism=100  

### Jupyter should have launched and you should now be running the jupyter notebook with Spark and SystemML!


### Now set up the notebook and download the data:

%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np  
import matplotlib.pyplot as plt  
plt.rcParams['figure.figsize'] = (10, 6)

sc.addPyFile("https://raw.githubusercontent.com/apache/incubator-systemml/3d5f9b11741f6d6ecc6af7cbaa1069cde32be838/src/main/java/org/apache/sysml/api/python/SystemML.py")

%%sh

curl -O http://snap.stanford.edu/data/amazon0601.txt.gz  
gunzip amazon0601.txt.gz  

### Use PySpark to load the data into the Spark Data Frame

import pyspark.sql.functions as F  
dataPath = "amazon0601.txt"

X_train = (sc.textFile(dataPath)  
    .filter(lambda l: not l.startswith("#"))
    .map(lambda l: l.split("\t"))
    .map(lambda prods: (int(prods[0]), int(prods[1]), 1.0))
    .toDF(("prod_i", "prod_j", "x_ij"))
    .filter("prod_i < 500 AND prod_j < 500")
    .cache())

max_prod_i = X_train.select(F.max("prod_i")).first()[0]  
max_prod_j = X_train.select(F.max("prod_j")).first()[0]  
numProducts = max(max_prod_i, max_prod_j) + 1  
print("Total number of products: {}".format(numProducts))  

### Create a SystemML Context Object

from SystemML import MLContext  
ml = MLContext(sc)  

### Define a kernel for Poisson nonnegative matrix factorization (PNMF) in DML

pnmf = """  
X = read($X)  
X = X+1  
V = table(X[,1], X[,2])  
size = ifdef($size, -1)  
if(size > -1) {  
    V = V[1:size,1:size]
}
max_iteration = as.integer($maxiter)  
rank = as.integer($rank)

n = nrow(V)  
m = ncol(V)  
range = 0.01  
W = Rand(rows=n, cols=rank, min=0, max=range, pdf="uniform")  
H = Rand(rows=rank, cols=m, min=0, max=range, pdf="uniform")  
losses = matrix(0, rows=max_iteration, cols=1)  

### Run PNMF

i=1  
while(i <= max_iteration) {

  H = (H * (t(W) %*% (V/(W%*%H))))/t(colSums(W)) 
  W = (W * ((V/(W%*%H)) %*% t(H)))/t(rowSums(H))


  losses[i,] = -1 * (sum(V*log(W%*%H)) - as.scalar(colSums(W)%*%rowSums(H)))
  i = i + 1;
}

write(losses, $lossout)  
write(W, $Wout)  
write(H, $Hout)  
"""

### Execute the Algorithm

ml.reset()  
outputs = ml.executeScript(pnmf, {"X": X_train, "maxiter": 100, "rank": 10}, ["W", "H", "losses"])  

### Retrieve the Losses and Plot Them

losses = outputs.getDF(sqlContext, "losses")  
xy = losses.sort(losses.ID).map(lambda r: (r[0], r[1])).collect()  
x, y = zip(*xy)  
plt.plot(x, y)  
plt.xlabel('Iteration')  
plt.ylabel('Loss')  
plt.title('PNMF Training Loss')  

## Congratulations! You just ran SystemML!


 Copyright 2017 IBM Corp. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
