## Reversing Game of Life v2

This is more straightforward way of solving this problem. For a given field, we create a window for each cell,
and train binary classifier (DecisionTree\RandomForest in our case).

The best score achieved with this approach was 0.12946.
For comparison, Pure CNN gives 0.10904.