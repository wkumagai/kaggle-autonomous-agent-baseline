## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: ordinal
    * `feature_1`: ordinal
    * `feature_2`: count
    * `feature_3`: ordinal
    * `feature_4`: numeric
    * `feature_5`: ordinal
    * `feature_6`: numeric
    * `feature_7`: categorical
    * `feature_8`: count
    * `feature_9`: ordinal
    * `feature_10`: ordinal
    * `feature_11`: ordinal
    * `feature_12`: numeric
    * `feature_13`: ordinal
    * `feature_14`: categorical
    * `feature_15`: numeric
    * `feature_16`: categorical
    * `feature_17`: categorical
    * `feature_18`: numeric
    * `feature_19`: ordinal
    * `feature_20`: ordinal
    * `feature_21`: numeric
    * `feature_22`: ordinal
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
