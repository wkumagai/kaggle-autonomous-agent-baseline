## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: ordinal
    * `feature_1`: categorical
    * `feature_2`: ordinal
    * `feature_3`: ordinal
    * `feature_4`: numeric
    * `feature_5`: ordinal
    * `feature_6`: count
    * `feature_7`: numeric
    * `feature_8`: categorical
    * `feature_9`: numeric
    * `feature_10`: ordinal
    * `feature_11`: ordinal
    * `feature_12`: ordinal
    * `feature_13`: ordinal
    * `feature_14`: categorical
    * `feature_15`: ordinal
    * `feature_16`: ordinal
    * `feature_17`: numeric
    * `feature_18`: categorical
    * `feature_19`: categorical
    * `feature_20`: ordinal
    * `feature_21`: ordinal
    * `feature_22`: ordinal
    * `feature_23`: numeric
    * `feature_24`: ordinal
    * `feature_25`: ordinal
    * `feature_26`: categorical
    * `feature_27`: categorical
    * `feature_28`: ordinal
    * `feature_29`: ordinal
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
