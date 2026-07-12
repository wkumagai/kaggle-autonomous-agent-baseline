## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: count
    * `feature_1`: count
    * `feature_2`: numeric
    * `feature_3`: numeric
    * `feature_4`: ordinal
    * `feature_5`: count
    * `feature_6`: ordinal
    * `feature_7`: count
    * `feature_8`: count
    * `feature_9`: numeric
    * `feature_10`: count
    * `feature_11`: numeric
    * `feature_12`: ordinal
    * `feature_13`: numeric
    * `feature_14`: numeric
    * `feature_15`: numeric
    * `feature_16`: ordinal
    * `feature_17`: count
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
