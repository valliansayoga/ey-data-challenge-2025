import pandas as pd
from src.training import load_data


def create_submission(
    filename: str, model, submission_template_path: str, X_train: pd.DataFrame
) -> None:
    """
    Create a competition submission file using a trained model.

    Parameters
    ----------
    filename : str
        Path where the submission file will be saved.
    model : sklearn.base.BaseEstimator
        Trained model that implements the predict() method.
    submission_template_path : str
        Path to the submission template file containing Latitude and Longitude columns.
    X_train : pd.DataFrame
        Training data used to ensure consistent columns between training and prediction.

    Returns
    -------
    None
        The function saves the submission file to disk but doesn't return anything.

    Notes
    -----
    The function performs the following steps:
    1. Loads submission data using load_data()
    2. Reads the submission template
    3. Selects matching columns from submission data
    4. Makes predictions using the model
    5. Saves predictions to CSV file
    """
    sub_df = load_data("Submission")
    final_df = pd.read_csv(
        submission_template_path,
        usecols=["Latitude", "Longitude"],
    )

    print("Predicting", sub_df.shape[0], "rows...")
    to_predict = sub_df.loc[:, X_train.columns]
    to_predict.to_csv("validation_set.csv", index=False)
    final_df["UHI Index"] = model.predict(to_predict)
    final_df.to_csv(filename, index=False)
    print("Done!")
    return
