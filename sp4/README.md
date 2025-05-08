# README

[µMAG Standard Problem #4](https://www.ctcms.nist.gov/~rdm/std4/spec4.html)

## Exercise 


### Specifications
Standard problem #4 is focused on dynamic aspects of micromagnetic computations. The initial state is an equilibrium 
s-state such as is obtained after applying and slowly reducing a saturating field along the [1,1,1] direction to zero. 
Fields of magnitude sufficient to reverse the magnetization of the rectangle are applied to this initial state and the 
time evolution of the magnetization as the system moves towards equilibrium in the new fields are examined. 
The problem will be run for two different applied fields.

## Desired output for comparison: 

Two outputs are desired for comparison:

1. The (x,y,z) components of the spatially averaged magnetization of the sample as a function of time from t=0 until 
the sample reaches equilibrium in the new field. 
2. An image of the magnetization at the time when the x-component of the spatially averaged magnetization first 
crosses zero.

The magnetization values in the time series data should be normalized by Ms. 
The time series data is desired so that a detailed comparison can be made between solutions. 
The magnetization images are to check for any differences in the reversal mechanisms if the time data between 
solutions is different.
