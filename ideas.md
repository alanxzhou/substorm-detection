# Ideas for Improvement

There is a good chance that using solar wind data / Interplanetary Magnetic Field (IMF) data will increase our prediction capability. 

Right now we have a CNN acting on individual stations and points are evaluated based on the closest 5 stations, these stations are 
combined in 1 step using a [5x1xC] kernel stations x time x channels. Is this the best architecture?

- extra sources of data
    - solar wind data is very patchy, need a way to fill in the gaps in order to use it
    - make a CNN or some other architecure to process the sequence
    - figure out joint mag - solar wind architecture
- architecture
    - use RNN for the individual stations' data
    - use RNN to combine variable number of stations?
    - use graph NN type combination strategy to hierarchichally combine nearby stations incrementally increasing effective-field-of-view
    - deep-sets-type permutation / size invariant combinations
    - any sort of sphere-aware convolutional layers
- inputs / data pipeline
    - possible data augmentation?
    - input sequences are currently non-overlapping, should they be?
    - training is slow because opening large .nc files is slow, can this be improved?
    - How should we even be training this thing? closest-station approach or just go through time
- outputs
    - binary label for SS occurring in next hour (or T minutes)
    - 2d location on earth
    - multi-class label (no storm, 0-15, 15-30, 30-45, 45-60) minutes out from storm
    - actually just predict a whole hours worth of SME / AE / some other index
    - use an RNN for output layers in order to have it able to predict 2 or more storms in the next hour or 3
- visualization
    - ultimately, if we can get good results, it would be very interesting to see what features the network has learned. 
      could inform physics?
    - 
- misc
    - need to be using [distance along earth](https://en.wikipedia.org/wiki/Great-circle_distance), not l2 norm of lat and lon
    - solve ai
