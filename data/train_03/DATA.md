## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: count
    * `feature_1`: categorical
    * `feature_2`: count
    * `feature_3`: count
    * `feature_4`: categorical
    * `feature_5`: categorical
    * `feature_6`: categorical
    * `feature_7`: numeric
    * `feature_8`: count
    * `feature_9`: categorical
    * `feature_10`: numeric
    * `feature_11`: count
    * `feature_12`: numeric
    * `feature_13`: count
    * `feature_14`: count
    * `feature_15`: ordinal
    * `feature_16`: categorical
    * `feature_17`: count
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
