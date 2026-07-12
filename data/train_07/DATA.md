## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: categorical
    * `feature_1`: count
    * `feature_2`: numeric
    * `feature_3`: ordinal
    * `feature_4`: numeric
    * `feature_5`: ordinal
    * `feature_6`: ordinal
    * `feature_7`: ordinal
    * `feature_8`: count
    * `feature_9`: numeric
    * `feature_10`: count
    * `feature_11`: numeric
    * `feature_12`: count
    * `feature_13`: count
    * `feature_14`: count
    * `feature_15`: categorical
    * `feature_16`: numeric
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
