import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from Source.data import balanced_train_valid_split, selected_balanced_train_valid_test_split, root_mean_squared_error
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.global_poolings import MaxPooling
from config import ROOT_DIR

from I.am.sorry.but.it.is.secret.information.now import model_parameters


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

train_metals = 'Ba Ca Li Tl Hg Au Ac La Mg Zr Pb Cd Ag Pm Ce Cu'.split()
test_metals = ['Sr']

experiment_name = "multimetal_training_test_Sr"
cv_folds = 5
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
output_folder = ROOT_DIR / f"Output/{experiment_name}/{cv_folds}fold_{mode}_{time_mark}"


targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)


train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=str(train_sdf_folder / f"{metal}.sdf"),
                                                          mol_featurizer=ConvMolFeaturizer(),
                                                          metal_featurizer=SkipatomFeaturizer())
                  for metal in tqdm(train_metals, desc="Featurizig")]

test_dataset = [featurize_sdf_with_metal_and_conditions(path_to_sdf=str(train_sdf_folder / f"{metal}.sdf"),
                                                        mol_featurizer=ConvMolFeaturizer(),
                                                        metal_featurizer=SkipatomFeaturizer())
                for metal in tqdm(test_metals, desc="Featurizig")]

logging.info("Splitting...")
folds = balanced_train_valid_split(train_datasets,
                                   n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)

test = DataLoader(test_dataset[0], batch_size=len(test_dataset[0]), shuffle=False)

model = GCNN_FCNN(
    metal_features=next(iter(folds[0][0])).metal_x.shape[-1],
    node_features=next(iter(folds[0][0])).x.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    targets=targets,
    seed=seed,
)

trainer.train_cv_models()