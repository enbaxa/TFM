"""
This is just a quick package so a user can extract simple syntax
paste it in python interactive and instantly be able to tinker with
the api, a loaded dataset and its corresponding trained model.

The dataset, like all others is provided in test_codes/test_datasets
The trained model is simply named "model" and is in test_codes/out/medium_3_l1.
This is because it corresponds to the medium_3 dataset.
The user can see both the model.ini in the directory and model.pth.
Both are loaded in the snippet below

The dataset is loaded as a pandas DataFrame and the model is loaded as a torch model.

The user is assumed to be running python interactive from the root level of the project.
This is important because the paths are relative, in case the user did not install the package.

After executing those lines, the user has acces to:
    - df: a pandas DataFrame with the dataset
    - model: a torch model with the trained model
    - api: the model api object

The user can now run any of the api functions on the model and dataset.
The user can also run the execute function on the model to get predictions.

For example:
    model.execute("addConnection", mode="priority")
    model.execute("addConnection", mode="multilabel")

Any input will produce an output, but if it is not in the dataset,
the output is just a random guess, so the user can see the model is working.
"""

import pandas as pd
import model_api

def print_guesses(model, inp, i=6):
    print(f"Input: {inp}")
    print(f"Multilabel guess: {model.execute(inp, mode='multilabel')}")
    top_guesses = "\n".join(("\n", *model.execute(inp, mode='priority')[:i]))
    print(f"Priority guess: {top_guesses}")

api = model_api.ModelApi()
df = pd.read_csv("test_codes/test_datasets/medium_3.csv")
model = api.load_model("test_codes/out/medium_3_l1/model")

