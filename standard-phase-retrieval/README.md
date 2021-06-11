
# Code for Standard phase retrieval section in the paper
## Algorithms Implemented:

**Model BPG**: Model based Bregman Proximal Gradient (BPG)   
**Model BPG-WB**: Model based BPG with Backtracking  
**IBPM-LS**: Inexact Bregman Proximal Minimization Line Search Algorithm

### Dependencies
- numpy, matplotlib

If you have installed above mentioned packages you can skip this step. Otherwise run  (maybe in a virtual environment):

    pip install -r requirements.txt

## Reproduce results

To generate results 

    chmod +x generate_results.sh
    ./generate_results.sh

Then to create the plots
    
    chmod +x generate_plots.sh
    ./generate_plots.sh

Now you can check **figures** folder for various figures. 

## Description of results

The function number is denoted as **fun_num**. 

In **fun_num**  1 : L1-Regularization is used.  
In **fun_num**  2 : Squared L2-Regularization is used.  

