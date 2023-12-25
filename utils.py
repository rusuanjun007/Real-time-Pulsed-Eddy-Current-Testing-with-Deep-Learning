import glob
import os
import pickle
from os.path import join
from typing import Callable
from urllib.parse import unquote, urlparse

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from mlflow.entities import RunStatus
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.logging_utils import eprint


def _already_ran(hyperparams, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.search_runs(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.info.run_id)
        tags = full_run.data.tags
        match_failed = False
        for param_key, param_value in hyperparams.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != str(param_value):
                match_failed = True
                break
        if match_failed:
            continue

        if "t50851tm" not in unquote(urlparse(full_run.info.artifact_uri).path):
            if (
                len(
                    glob.glob(
                        join(
                            unquote(urlparse(full_run.info.artifact_uri).path), "Epoch*"
                        )
                    )
                )
                != 0
            ):
                starting_epoch = max(
                    [
                        int(os.path.normpath(p).split(os.sep)[-1][5:])
                        for p in glob.glob(
                            join(
                                unquote(urlparse(full_run.info.artifact_uri).path),
                                "Epoch*",
                            )
                        )
                    ]
                )
            else:
                starting_epoch = 0
        else:
            if (
                glob.glob(
                    join(
                        os.path.relpath(
                            unquote(urlparse(full_run.info.artifact_uri).path),
                            "/net/scratch2/t50851tm/momaml_jax",
                        ),
                        "Epoch*",
                    )
                )
                != 0
            ):
                starting_epoch = max(
                    [
                        int(os.path.normpath(p).split(os.sep)[-1][5:])
                        for p in glob.glob(
                            join(
                                os.path.relpath(
                                    unquote(urlparse(full_run.info.artifact_uri).path),
                                    "/net/scratch2/t50851tm/momaml_jax",
                                ),
                                "Epoch*",
                            )
                        )
                    ]
                )
            else:
                starting_epoch = 0
        if run_info.to_proto().info.status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED" "(run_id=%s, status=%s)")
                % (run_info.info.run_id, run_info.info.status)
            )
            return (False, run_info.info.run_id, starting_epoch)
        eprint(
            ("Found matching run and it has already finished." "(run_id=%s, status=%s)")
            % (run_info.info.run_id, run_info.info.status)
        )
        return (True, run_info.info.run_id, starting_epoch)
    eprint("No matching run has been found.")
    return (False, None, 0)


def mf_loghyperparams(hparams, experiment):
    for k in hparams.keys():
        if k == "number_of_experiment":
            continue
        vvv = str(hparams[k][experiment])
        if len(vvv) > 500:
            vvv = vvv[0:500]
        mlflow.log_param(k, vvv)


def save_data(ckpt_dir, data_dict, dataname):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # Save data.
    with open(os.path.join(ckpt_dir, dataname + "_array.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(data_dict):
            np.save(f, x, allow_pickle=False)
    # Save structure of data.
    tree_struct = jax.tree_map(lambda t: 0, data_dict)
    with open(os.path.join(ckpt_dir, dataname + "_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)
    print(dataname + " saved.")


def restore(ckpt_dir, dataname):
    with open(os.path.join(ckpt_dir, dataname + "_tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, dataname + "_array.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]
    print("restore " + dataname)
    return jax.tree_util.tree_unflatten(treedef, flat_state)


def lr_schedule(lr: float, lr_schedule_flag: bool, num_update_step: int) -> Callable:
    # Define learning rate schedule.
    if lr_schedule_flag:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr / 3.0,
            peak_value=lr,
            warmup_steps=int(num_update_step / 5),
            decay_steps=int(num_update_step),
            end_value=0.0,
        )
    else:
        schedule = optax.constant_schedule(lr)
    return schedule


def optimizerSelector(opName: str) -> optax.GradientTransformation:
    optDict = {
        "adabelief": optax.adabelief,
        "adafactor": optax.adafactor,
        "adagrad": optax.adagrad,
        "adam": optax.adam,
        "adamw": optax.adamw,
        "fromage": optax.fromage,
        "lamb": optax.lamb,
        "lars": optax.lars,
        "noisy_sgd": optax.noisy_sgd,
        "dpsgd": optax.dpsgd,
        "radam": optax.radam,
        "rmsprop": optax.rmsprop,
        "sgd": optax.sgd,
        "sm3": optax.sm3,
        "yogi": optax.yogi,
    }
    if opName in optDict.keys():
        print(f"Optimizer {opName} is selected.")
        return optDict[opName]
    else:
        raise ValueError(f"No optimizer named {opName}.")


def lossSelector(lossName: str) -> Callable:
    def msle(true, pred):
        return 0.5 * jnp.mean((jnp.log(true + 1) - jnp.log(pred + 1)) ** 2)

    lossDict = {
        "cosine_distance": optax.cosine_distance,
        "l2_loss": optax.l2_loss,
        "softmax_cross_entropy": optax.softmax_cross_entropy,
        "huber_loss": optax.huber_loss,
        "msle": msle,
    }
    if lossName in lossDict.keys():
        print(f"Loss {lossName} is selected.")
        return lossDict[lossName]
    else:
        raise ValueError(f"No loss named {lossName}.")


def weightDecay(params: dict) -> jnp.ndarray:
    decayLoss = jnp.array(0.0, dtype=jnp.float32)
    for layerName in params.keys():
        for paramName in params[layerName].keys():
            if "batch_norm" not in layerName:
                decayLoss += optax.l2_loss(params[layerName][paramName]).sum()
    return decayLoss


def calculate_norm(param_dict):
    flatten_dict = jax.tree_util.tree_flatten(param_dict)[0]
    result = jnp.array(0.0, dtype=jnp.float32)
    for one_data in flatten_dict:
        result += jax.numpy.linalg.norm(one_data)
    return result
