## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: categorical
    * `feature_1`: numeric
    * `feature_2`: numeric
    * `feature_3`: numeric
    * `feature_4`: categorical
    * `feature_5`: numeric
    * `feature_6`: categorical
    * `feature_7`: numeric
    * `feature_8`: ordinal
    * `feature_9`: numeric
    * `feature_10`: numeric
    * `feature_11`: categorical
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
