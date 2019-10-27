## Reversing Game of Life v2

This is more straightforward way of solving this problem.
<br />
For a given field, we create a window for each cell of its direct neighbors,
and train binary classifier (DecisionTree\RandomForest in our case) to predict central cell's state.
<br />
The best score achieved with this approach was 0.12946.
For comparison, Pure CNN gives 0.10904.

## Bonuses

* Optimized code for base model (Cython)
* Verbose mode
* Advanced algorithm
* CNN result in top-5 on private LB

## FAQ

Q: When I include FastDecisionTree, error occurs with "clang error: #include numpy/array.h not found"
A: `export CFLAGS="-I /Users/ptyshevs/goinfre/env/lib/python3.7/dist-packages/numpy/core/include/ $CFLAGS"`
