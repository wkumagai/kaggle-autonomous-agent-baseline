## Files ##

* **train.csv**: the training set
* **test.csv**: the test set
* **sample_submission.csv**: a submission file in the correct format

## Columns ##

* **{train/test}.csv**
    * `row_id`: a unique identifier for this row
    * `feature_0`: categorical
    * `feature_1`: categorical
    * `feature_2`: categorical
    * `feature_3`: categorical
    * `feature_4`: categorical
    * `feature_5`: categorical
    * `feature_6`: categorical
    * `feature_7`: categorical
    * `feature_8`: categorical
    * `target`: binary categorical, the target, only in `train.csv`

* **sample_submission.csv**
    * `row_id`: corresponding to the `row_id` in `test.csv`
    * `target`: the target for each row of the test set
