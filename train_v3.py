import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import os
from os.path import exists, join
from typing import Union, List, NamedTuple, Callable, Tuple
import numpy as np
import haiku as hk
import multiprocessing
import mlflow
from urllib.parse import unquote, urlparse
import time
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import graphviz
import json
import copy


import dataset_v2
import visualization
import utils
import model
from jax_helper.utils.visualization import visualize_tf_model
from jax_helper.utils.tflite_tools import (
    convert_hk_to_tf,
    convert_tf_to_tflite,
    run_tflite_model,
    summary_tf_model,
)


class expState(NamedTuple):
    """
    diff_model_state = {params: hk.Params}
    non_diff_state = {state: hk.State
                      optState: optax.OptState}
    """

    diff: dict
    non_diff: dict


def define_forward(hparams: dict, experiment: int) -> Callable:
    return model.hk_PEC_model(hparams, experiment)


def define_loss_fn(
    forward: hk.Transformed,
    is_training: bool,
    optax_loss: Callable,
    hparams: dict,
    experiment: int,
) -> Callable:
    @jax.jit
    def loss_fn(
        params: hk.Params, state: hk.State, data_dict: dict
    ) -> Tuple[jnp.ndarray, Tuple[hk.State, jnp.ndarray]]:
        # Forward-pass.
        if is_training:
            # Update state.
            y_pred, state = forward.apply(params, state, data_dict["data"], is_training)
        else:
            # Do not update state.
            y_pred, _ = forward.apply(params, state, data_dict["data"], is_training)

        # Calculate loss. loss = target_loss + a * weight_decay.
        loss = 0.0
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            # Calculate mean loss.
            if "regression" in hparams["problem"][experiment]:
                # If regression, true label is y_pred[pred_head_key].
                y_pred[head_name] = y_pred[head_name].reshape(-1)
                loss += optax_loss(y_pred[head_name], data_dict[head_name]).mean()
            elif "classification" in hparams["problem"][experiment]:
                # If classification, true label is data_dict[head_name + "Label"]
                loss += optax_loss(
                    y_pred[head_name], data_dict[head_name + "Label"]
                ).mean()

        # Average multi-heads loss.
        loss = loss / len(hparams["multi_head_instruction"][experiment])

        # Add weight decay.
        if hparams["weight_decay"][experiment] is not None:
            decayLoss = hparams["weight_decay"][experiment] * utils.weightDecay(params)
            loss += decayLoss
        return loss, (state, y_pred)

    return loss_fn


def define_train_step(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    optimizer_schedule: Callable,
) -> Callable:
    @jax.jit
    def train_step(
        train_exp_state: expState, data_dict: dict
    ) -> Tuple[expState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Forward-pass and backward-pass.
        (
            (loss, (train_exp_state.non_diff["state"], y_pred)),
            grads_dict,
        ) = jax.value_and_grad(loss_fn, has_aux=True)(
            train_exp_state.diff["params"], train_exp_state.non_diff["state"], data_dict
        )

        # Record inner learning rate.
        record_lr = optimizer_schedule(train_exp_state.non_diff["opt_state"][0].count)

        # Calculate gradient_update and update opt_state.
        updates, train_exp_state.non_diff["opt_state"] = optimizer.update(
            grads_dict, train_exp_state.non_diff["opt_state"]
        )

        # update params.
        train_exp_state.diff["params"] = optax.apply_updates(
            train_exp_state.diff["params"], updates
        )
        return train_exp_state, loss, y_pred, record_lr, grads_dict

    return train_step


def discard_str_keys(record, data_dict):
    """
    Discard str keys in data_dict and save fileName as identifier.
    """
    for data_key in list(data_dict.keys()):
        if data_dict[data_key].dtype == object:
            if data_key == "fileName":
                if data_key not in record.keys():
                    record[data_key] = []
                temp_data = data_dict[data_key].copy().tolist()
                record[data_key].append([d.decode() for d in temp_data])
            del data_dict[data_key]


def define_train(train_step: Callable, hparams: dict, experiment: int) -> Callable:
    def train(
        train_exp_state: expState,
        dataset: tf.data.Dataset,
    ) -> Tuple[expState, dict]:
        record = {"loss": [], "lr": [], "grads_norm": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Update train_exp_state.
            train_exp_state, loss, y_pred, lr, grads_dict = train_step(
                train_exp_state, data_dict
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            record["lr"].append(lr.tolist())
            record["grads_norm"].append(utils.calculate_norm(grads_dict).tolist())
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "regression" == hparams["problem"][experiment]:
                true_value = np.concatenate(record[head_name])
                pred_value = np.concatenate([r[head_name] for r in record["y_pred"]])
                mean_abs_error = np.mean(np.abs(true_value - pred_value))
                mean_ralative_error = np.mean(
                    np.abs(true_value - pred_value) / (true_value + 0.1)
                )
                record["mean_abs_error"] = mean_abs_error.item()
                record["mean_ralative_error"] = mean_ralative_error.item()

        return train_exp_state, record

    return train


def define_test(lossFn: Callable, hparams: dict, experiment: int) -> Callable:
    def test(text_exp_state: expState, dataset: tf.data.Dataset) -> dict:
        record = {"loss": [], "y_pred": []}
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "classification" == hparams["problem"][experiment]:
                record_name = head_name + "Label"
            elif "regression" == hparams["problem"][experiment]:
                record_name = head_name
            record[record_name] = []

        for data_dict in dataset.as_numpy_iterator():
            discard_str_keys(record, data_dict)
            # Do not update state.
            loss, (_, y_pred) = lossFn(
                text_exp_state.diff["params"],
                text_exp_state.non_diff["state"],
                data_dict,
            )
            record["loss"].append(loss.tolist())
            record["y_pred"].append({k: y_pred[k].tolist() for k in y_pred.keys()})
            for head_name in hparams["multi_head_instruction"][experiment].keys():
                if "classification" == hparams["problem"][experiment]:
                    record_name = head_name + "Label"
                elif "regression" == hparams["problem"][experiment]:
                    record_name = head_name
                record[record_name].append(data_dict[record_name].tolist())

        record["loss"] = jnp.mean(jnp.array(record["loss"])).tolist()
        for head_name in hparams["multi_head_instruction"][experiment].keys():
            if "regression" == hparams["problem"][experiment]:
                true_value = np.concatenate(record[head_name])
                pred_value = np.concatenate([r[head_name] for r in record["y_pred"]])
                mean_abs_error = np.mean(np.abs(true_value - pred_value))
                mean_ralative_error = np.mean(
                    np.abs(true_value - pred_value) / (true_value + 0.1)
                )
                record["mean_abs_error"] = mean_abs_error.item()
                record["mean_ralative_error"] = mean_ralative_error.item()

        return record

    return test


def define_forward_and_optimizer(
    hparams: dict, experiment: int, data_name: str, DATA_SIZE: Tuple
):
    # Define _forward.
    _forward = define_forward(hparams, experiment)

    # Define optimizer and learning rate schedule.
    nData = len(glob.glob(join(DATA_ROOT, data_name, "*")))

    optimizerSchedule = utils.lr_schedule(
        hparams["lr"][experiment],
        hparams["lr_schedule_flag"][experiment],
        int(
            nData
            * hparams["split_ratio"][experiment]
            * hparams["epoch"][experiment]
            / hparams["batch_size"][experiment]
        ),
    )
    optimizer = utils.optimizerSelector(hparams["optimizer"][experiment])(
        learning_rate=optimizerSchedule
    )

    def summary_model():
        """
        Summary model.
        """

        def temp_forward(x):
            _forward(x, True)

        # Summary model.
        dummy_x = np.random.uniform(
            size=(
                hparams["batch_size"][experiment],
                DATA_SIZE[0],
                DATA_SIZE[1],
                DATA_SIZE[2],
            )
        ).astype(np.float32)
        summary_message = f"{hk.experimental.tabulate(temp_forward)(dummy_x)}"
        return summary_message

    summary_message = summary_model()

    return (_forward, optimizer, optimizerSchedule, summary_message)


def initialize_train_exp_state(DATA_SIZE, forward, optimizer, mlflow_artifact_path):
    # Initialize the parameters and states of the network and return them.
    dummy_x = np.random.uniform(
        size=(1, DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2])
    ).astype(np.float32)
    params, state = forward.init(
        rng=jax.random.PRNGKey(42), x=dummy_x, is_training=True
    )

    # Visualize model.
    dot = hk.experimental.to_dot(forward.apply)(params, state, dummy_x, True)
    dot_plot = graphviz.Source(dot)
    dot_plot.source.replace("rankdir = TD", "rankdir = TB")
    dot_plot_save_path = join(mlflow_artifact_path, "summary")
    dot_plot.render(filename="model_plot", directory=dot_plot_save_path)

    # Initialize model and optimiser.
    opt_state = optimizer.init(params)

    # Initialize train state.
    train_exp_state = expState(
        {"params": params}, {"state": state, "opt_state": opt_state}
    )
    return train_exp_state


def save_exp_state(exp_state, epoch, mlflow_artifact_path):
    saving_history = {
        int(os.path.normpath(p).split(os.sep)[-1][5:]): p
        for p in glob.glob(join(mlflow_artifact_path, "Epoch*"))
    }
    # Only keep the latest N models.
    if len(saving_history) >= 5:
        shutil.rmtree(saving_history[min(saving_history.keys())])
        print("Delete save", min(saving_history.keys()))

    save_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(epoch))
    utils.save_data(save_ckpt_dir, exp_state.diff["params"], "params")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["state"], "state")
    utils.save_data(save_ckpt_dir, exp_state.non_diff["opt_state"], "opt_state")


def restore_exp_state(starting_epoch, mlflow_artifact_path):
    restore_ckpt_dir = join(mlflow_artifact_path, "Epoch" + str(starting_epoch))
    print(f"Restore from {restore_ckpt_dir}")

    params = utils.restore(restore_ckpt_dir, "params")
    state = utils.restore(restore_ckpt_dir, "state")
    opt_state = utils.restore(restore_ckpt_dir, "opt_state")

    exp_state = expState({"params": params}, {"state": state, "opt_state": opt_state})
    return exp_state


def plot_confusion_matrix(
    train_result: dict,
    val_result: dict,
    test_result: dict,
    train_dataset_dict: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent: bool = False,
):
    def total_mean_acc(confusionMatrix):
        cmShape = confusionMatrix.shape
        cnt = 0
        for ii in range(cmShape[0]):
            cnt += confusionMatrix[ii][ii]
        return cnt / np.sum(confusionMatrix)

    labelCodeBook = {}
    for head_name in hparams["multi_head_instruction"][experiment].keys():
        temp_list = list(
            set(
                train_dataset_dict[data_key]["metaDataAndLabel"][head_name]
                for data_key in train_dataset_dict.keys()
            )
        )
        temp_list.sort()
        labelCodeBook[head_name] = temp_list

    fig = plt.figure(
        figsize=(4 + 4 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 1 + len(hparams["multi_head_instruction"][experiment]), figure=fig
    )

    for nR, (name, result) in enumerate(
        zip(["Train", "validation", "Test"], [train_result, val_result, test_result])
    ):
        text_info = f"Total Mean Acc: \n"
        for nH, head_name in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):
            dataTrue = np.argmax(np.concatenate(result[head_name + "Label"]), axis=1)
            dataPred = np.argmax(
                np.concatenate([re[head_name] for re in result["y_pred"]]), axis=1
            )
            dataCm = tf.math.confusion_matrix(
                dataTrue,
                dataPred,
                num_classes=np.max(dataTrue) + 1,
            ).numpy()
            # Plot train confusion matrix.
            dataAxes = fig.add_subplot(gs[nR, nH])
            sns.heatmap(dataCm, annot=True, fmt="d", ax=dataAxes, cmap="YlGnBu")

            dataAxes.set_title(name + " " + head_name, wrap=True)
            dataAxes.set_xlabel("Pred Labels")
            dataAxes.set_ylabel("True Labels")
            dataAxes.set_xticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )
            dataAxes.set_yticklabels(
                labelCodeBook[head_name], fontsize=6.5, rotation=np.pi / 4
            )

            text_info += f"{head_name}: {total_mean_acc(dataCm):.3f}\n"

        textAxes = fig.add_subplot(gs[nR, nH + 1])
        textAxes.axis("off")
        textAxes.text(0, 0.3, text_info, wrap=True)

    # Set fig property.
    plt.tight_layout()

    figSavePath = join(
        mlflowArtifactPath, "confusion_matrix", "Epoch" + str(starting_epoch)
    )
    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "confusion matrix.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)


def plot_residual_fig2(
    trainResult: dict,
    val_result: dict,
    testResult: dict,
    hparams: dict,
    experiment: int,
    starting_epoch: int,
    mlflowArtifactPath: str,
    transparent=False,
):
    FONTSIZE = 17

    def createTable(result, headType):
        headClass = sorted(
            list(set(np.concatenate(result[headType]).astype(np.int32).tolist()))
        )
        classStatistic = {hC: {"Name": str(hC) + " mm"} for hC in headClass}
        for classKey in classStatistic.keys():
            dataTrue = np.concatenate(result[headType])
            dataPred = np.concatenate([re[headType] for re in result["y_pred"]])
            dataIndex = np.where(dataTrue == classKey)
            classStatistic[classKey]["True"] = dataTrue[dataIndex]
            classStatistic[classKey]["Pred"] = dataPred[dataIndex]
            classStatistic[classKey]["Abs Error Mean"] = np.mean(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Abs Error Std"] = np.std(
                np.abs(dataTrue[dataIndex] - dataPred[dataIndex])
            )
            classStatistic[classKey]["Amount"] = len(dataTrue[dataIndex])
        tableReturn = {}
        tableReturn["colLabels"] = [
            classStatistic[k]["Name"] for k in classStatistic.keys()
        ]
        tableReturn["rowLabels"] = ["Amount", "Abs Error Mean", "Abs Error Std"]
        tableReturn["cellText"] = [
            ["%d" % classStatistic[k]["Amount"] for k in classStatistic.keys()]
        ]
        for tables_rows in ["Abs Error Mean", "Abs Error Std"]:
            tableReturn["cellText"].append(
                ["%.3f" % classStatistic[k][tables_rows] for k in classStatistic.keys()]
            )

        return tableReturn, classStatistic

    fig = plt.figure(
        figsize=(5 + 5 * len(hparams["multi_head_instruction"][experiment]), 12)
    )
    gs = plt.GridSpec(
        3, 2 * len(hparams["multi_head_instruction"][experiment]), figure=fig
    )
    mean_err_dict = {}

    for nR, (name, result) in enumerate(
        zip(["Train", "Validation", "Test"], [trainResult, val_result, testResult])
    ):
        for nH, headType in enumerate(
            hparams["multi_head_instruction"][experiment].keys()
        ):
            tableReturn, classStatistic = createTable(result, headType)

            mean_err = 0
            mean_std = 0
            n_amount = 0
            for n_class in range(len(tableReturn["colLabels"])):
                n_amount += float(tableReturn["cellText"][0][n_class])
                mean_err += float(tableReturn["cellText"][0][n_class]) * float(
                    tableReturn["cellText"][1][n_class]
                )
                mean_std += float(tableReturn["cellText"][0][n_class]) * float(
                    tableReturn["cellText"][2][n_class]
                )
            mean_err_dict[name] = {
                "err": mean_err / n_amount,
                "std": mean_std / n_amount,
            }

            # Residual plots.
            residualAxes = fig.add_subplot(gs[nR, 3 * nH])
            tableAxes = fig.add_subplot(gs[nR, 3 * nH + 1])
            residualAxes.violinplot(
                [classStatistic[k]["Pred"] for k in classStatistic.keys()],
                list(classStatistic.keys()),
                widths=(list(classStatistic.keys())[1] - list(classStatistic.keys())[0])
                / 2,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )
            residualAxes.set_title(
                name + " Dataset Prediction Result", wrap=True, fontsize=FONTSIZE
            )
            residualAxes.set_xlabel("True " + headType + " (mm)", fontsize=FONTSIZE)
            residualAxes.set_ylabel(
                "Predicted " + headType + " (mm)", fontsize=FONTSIZE
            )

            table = tableAxes.table(
                cellText=tableReturn["cellText"],
                rowLabels=tableReturn["rowLabels"],
                colLabels=tableReturn["colLabels"],
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(FONTSIZE - 3)
            tableAxes.axis("off")
            tableAxes.set_title(name + " Dataset Summary", wrap=True, fontsize=FONTSIZE)

    figSavePath = join(
        mlflowArtifactPath, "residual_plot_new", "Epoch" + str(starting_epoch)
    )
    # Set fig property.
    plt.tight_layout()

    # Save fig.
    if not exists(figSavePath):
        os.makedirs(figSavePath)
        print(f"Create {figSavePath} to store image.png")
    fig.savefig(join(figSavePath, "residual plot.png"), transparent=transparent)

    # Close fig to release memory.
    # RuntimeWarning: More than 20 figures have been opened.
    # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory.
    # (To control this warning, see the rcParam `figure.max_open_warning`).
    plt.close(fig)
    return mean_err_dict


def convert_to_tflite(
    hparams, experiment, forward, params, state, DATA_SIZE, tflite_save_path
):
    # Create TF model.
    x = np.random.rand(16, DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]).astype(
        dtype=np.float32
    )
    tf_model = model.tf_PEC_model(DATA_SIZE, hparams, experiment)
    tf_outs = tf_model(x, training=True)
    for key_name in tf_outs.keys():
        print("TF model output:", key_name, tf_outs[key_name].shape)

    # Convert Haiku model to TF model.
    convert_hk_to_tf(params, state, tf_model)

    dataset_json_path = join(
        join(DATA_ROOT, "formatted_v2"),
        hparams["dataset_name"][experiment] + ".json",
    )

    # Create representative dataset for post-quantization.
    def create_representative_dataset():
        # Load dataset.
        train_dataset, _, _ = dataset_v2.dataPipeline(
            dataset_json_path,
            splitRate=hparams["split_ratio"][experiment],
            batchSize=1000,
            start_index=hparams["data_start_index"][experiment],
            n_samples=hparams["n_samples"][experiment],
            z_norm_flag=hparams["z_norm_flag"][experiment],
        )

        for dataDict in train_dataset.as_numpy_iterator():
            representative_dataset = dataDict["data"]
            break
        return representative_dataset

    # Convert TF model to TFLite model.
    uint8_tflite_path = join(
        tflite_save_path, "uint8" + hparams["model_name"][experiment] + ".tflite"
    )
    float32_tf_path = join(
        tflite_save_path, "float32" + hparams["model_name"][experiment] + ".tflite"
    )
    tflite_save_path = join(
        tflite_save_path, hparams["model_name"][experiment] + ".tflite"
    )

    convert_tf_to_tflite(
        tf_model,
        tflite_save_path,
        create_representative_dataset(),
        convert_mode="float32",
    )
    convert_tf_to_tflite(
        tf_model,
        tflite_save_path,
        create_representative_dataset(),
        convert_mode="uint8",
    )

    # Test Haiku, TF, TFLite model.
    train_dataset, _, test_dataset = dataset_v2.dataPipeline(
        dataset_json_path,
        splitRate=hparams["split_ratio"][experiment],
        batchSize=64,
        start_index=hparams["data_start_index"][experiment],
        n_samples=hparams["n_samples"][experiment],
        z_norm_flag=hparams["z_norm_flag"][experiment],
    )

    def test_acc(dataset, result_name: str):
        true_all = []
        hk_all_outs = []
        tf_all_outs = []
        uint8_tflite_all_outs = []
        float32_tflite_all_outs = []

        for dataDict in dataset.as_numpy_iterator():
            x = dataDict["data"]
            hk_out, _ = forward.apply(params, state, x, False)
            tf_out = tf_model(x, training=False)
            uint8_tflite_out = run_tflite_model(uint8_tflite_path, x, silent_mode=True)
            float32_tflite_out = run_tflite_model(float32_tf_path, x, silent_mode=True)

            true_all.append(dataDict["Thickness"])
            hk_all_outs.append(hk_out["Thickness"])
            tf_all_outs.append(tf_out["Thickness"])
            uint8_tflite_all_outs.append(uint8_tflite_out[0])
            float32_tflite_all_outs.append(float32_tflite_out[0])

        true_all = np.concatenate(true_all, axis=0)
        hk_all_outs = np.concatenate(hk_all_outs, axis=0).reshape(-1)
        tf_all_outs = np.concatenate(tf_all_outs, axis=0).reshape(-1)
        uint8_tflite_all_outs = np.concatenate(uint8_tflite_all_outs, axis=0).reshape(
            -1
        )
        float32_tflite_all_outs = np.concatenate(
            float32_tflite_all_outs, axis=0
        ).reshape(-1)

        zero_true_position = np.where(true_all == 0)[0]
        true_all = np.delete(true_all, zero_true_position)
        hk_all_outs = np.delete(hk_all_outs, zero_true_position)
        tf_all_outs = np.delete(tf_all_outs, zero_true_position)
        uint8_tflite_all_outs = np.delete(uint8_tflite_all_outs, zero_true_position)
        float32_tflite_all_outs = np.delete(float32_tflite_all_outs, zero_true_position)

        true_to_hk_error = np.mean(np.abs(true_all - hk_all_outs))
        true_to_tf_error = np.mean(np.abs(true_all - tf_all_outs))
        true_to_uint8_tflite_error = np.mean(np.abs(true_all - uint8_tflite_all_outs))
        true_to_float32_tflite_error = np.mean(
            np.abs(true_all - float32_tflite_all_outs)
        )

        ralative_true_to_hk_error = np.mean(np.abs(true_all - hk_all_outs) / true_all)
        ralative_true_to_tf_error = np.mean(np.abs(true_all - tf_all_outs) / true_all)
        ralative_true_to_uint8_tflite_error = np.mean(
            np.abs(true_all - uint8_tflite_all_outs) / true_all
        )
        ralative_true_to_float32_tflite_error = np.mean(
            np.abs(true_all - float32_tflite_all_outs) / true_all
        )

        hk_to_tf_error = np.mean(np.abs(hk_all_outs - tf_all_outs))
        hk_to_uint8_tflite_error = np.mean(np.abs(hk_all_outs - uint8_tflite_all_outs))
        tf_to_uint8_tflite_error = np.mean(np.abs(tf_all_outs - uint8_tflite_all_outs))
        hk_to_float32_tflite_error = np.mean(
            np.abs(hk_all_outs - float32_tflite_all_outs)
        )
        tf_to_float32_tflite_error = np.mean(
            np.abs(tf_all_outs - float32_tflite_all_outs)
        )
        uint8_tflite_to_float32_tflite_error = np.mean(
            np.abs(uint8_tflite_all_outs - float32_tflite_all_outs)
        )

        result_dict = {
            "true_to_hk_error": f"{true_to_hk_error:.2f}",
            "true_to_tf_error": f"{true_to_tf_error:.2f}",
            "true_to_uint8_tflite_error": f"{true_to_uint8_tflite_error:.2f}",
            "true_to_float32_tflite_error": f"{true_to_float32_tflite_error:.2f}",
            "ralative_true_to_hk_error": f"{ralative_true_to_hk_error * 100:.2f}%",
            "ralative_true_to_tf_error": f"{ralative_true_to_tf_error * 100:.2f}%",
            "ralative_true_to_uint8_tflite_error": f"{ralative_true_to_uint8_tflite_error * 100:.2f}%",
            "ralative_true_to_float32_tflite_error": f"{ralative_true_to_float32_tflite_error * 100:.2f}%",
            "hk_to_tf_error": hk_to_tf_error.tolist(),
            "hk_to_uint8_tflite_error": hk_to_uint8_tflite_error.tolist(),
            "tf_to_uint8_tflite_error": tf_to_uint8_tflite_error.tolist(),
            "hk_to_float32_tflite_error": hk_to_float32_tflite_error.tolist(),
            "tf_to_float32_tflite_error": tf_to_float32_tflite_error.tolist(),
            "uint8_tflite_to_float32_tflite_error": uint8_tflite_to_float32_tflite_error.tolist(),
            "true_all": true_all.tolist(),
            "hk_all_outs": hk_all_outs.tolist(),
            "tf_all_outs": tf_all_outs.tolist(),
            "uint8_tflite_all_outs": uint8_tflite_all_outs.tolist(),
            "float32_tflite_all_outs": float32_tflite_all_outs.tolist(),
        }

        save_dir = "/".join(tflite_save_path.split("/")[:-1])
        with open(join(save_dir, result_name + ".json"), "w") as f:
            json.dump(result_dict, f)

    test_acc(train_dataset, "train_result_tflite_summary")
    test_acc(test_dataset, "test_result_tflite_summary")


def summary_result(data_dict, pred_result):
    summary_dict = {}

    def summary_label(label_name, true_values, thickness_pred):
        if label_name in true_values.keys():
            if label_name == "Thickness" and true_values[label_name] == 0:
                return
            if label_name not in summary_dict.keys():
                summary_dict[label_name] = {}
            if true_values[label_name] not in summary_dict[label_name].keys():
                summary_dict[label_name][true_values[label_name]] = {}
                summary_dict[label_name][true_values[label_name]]["true_thickness"] = []
                summary_dict[label_name][true_values[label_name]]["pred_thickness"] = []
            if true_values["Thickness"] != 0:
                summary_dict[label_name][true_values[label_name]][
                    "true_thickness"
                ].append(true_values["Thickness"])
                summary_dict[label_name][true_values[label_name]][
                    "pred_thickness"
                ].append(thickness_pred)

    for y_preds, file_names in zip(pred_result["y_pred"], pred_result["fileName"]):
        for thickness_pred, file_name in zip(y_preds["Thickness"], file_names):
            true_values = data_dict[file_name]["metaDataAndLabel"]
            summary_label("Thickness", true_values, thickness_pred)
            summary_label("Lift-off", true_values, thickness_pred)
            summary_label("Insulation", true_values, thickness_pred)
            summary_label("WeatherJacket", true_values, thickness_pred)
            summary_label("Loc", true_values, thickness_pred)

    ralaive_error_dict = {}

    for label_name in summary_dict.keys():
        ralaive_error_dict[label_name] = {}
        for key_names in summary_dict[label_name].keys():
            ralaive_error = np.mean(
                np.abs(
                    np.array(summary_dict[label_name][key_names]["true_thickness"])
                    - np.array(summary_dict[label_name][key_names]["pred_thickness"])
                )
                / np.array(summary_dict[label_name][key_names]["true_thickness"])
            )

            n_data = len(summary_dict[label_name][key_names]["true_thickness"])

            ralaive_error_dict[label_name][key_names] = (ralaive_error, n_data)

    for label_name in summary_dict.keys():
        mean_ralatice_error = 0.0
        data_amount = 0
        for key_names in summary_dict[label_name].keys():
            mean_ralatice_error += (
                ralaive_error_dict[label_name][key_names][0]
                * ralaive_error_dict[label_name][key_names][1]
            )
            data_amount += ralaive_error_dict[label_name][key_names][1]
        mean_ralatice_error /= data_amount

        ralaive_error_dict[label_name]["mean"] = f"{mean_ralatice_error*100:.2f}%"

        for key_names in summary_dict[label_name].keys():
            ralaive_error_dict[label_name][key_names] = (
                f"{ralaive_error_dict[label_name][key_names][0]*100:.2f}%",
                f"{ralaive_error_dict[label_name][key_names][1]:.2f}",
            )

    return ralaive_error_dict


def main(training_flag, DEVICE, data_root, convert_flag):
    # Get number of avaliable CPU cores.
    local_cpu_count = multiprocessing.cpu_count()
    # Get number of avaliable GPU.
    local_gpu_count = jax.local_device_count()
    print(f"-----Avaliable CPU cores: {local_cpu_count}, GPU: {local_gpu_count}-----")

    # Define log epoch.
    LOG_EPOCH = 10
    # Set mlflow parameters.
    mlflow.set_registry_uri(join(".", DEVICE))
    mlflow.set_tracking_uri(join(".", DEVICE))

    if "PECRuns" in DEVICE:
        hparams = {
            # -----------Experiment setting-----------
            "number_of_experiment": 14,
            "description": [
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on Q345 dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
                "This model is trained on aluminum dataset, train without 0 aligenment.",
            ],
            "run_name": [
                "steel_lightweight_CNN",
                "steel_MLP",
                "steel_CNN",
                "steel_ResNet18-1D",
                "steel_ResNet18-1D_v2",
                "steel_ResNet34-1D",
                "steel_ResNet34-1D_v2",
                "aluminum_lightweight_CNN",
                "aluminum_MLP",
                "aluminum_CNN",
                "aluminum_ResNet18-1D",
                "aluminum_ResNet18-1D_v2",
                "aluminum_ResNet34-1D",
                "aluminum_ResNet34-1D_v2",
            ],
            # -----------Model setting-----------
            "model_name": [
                "simpleCNN",
                "mlp",
                "simpleCNN",
                "resnet18",
                "resnet18",
                "resnet34",
                "resnet34",
                "simpleCNN",
                "mlp",
                "simpleCNN",
                "resnet18",
                "resnet18",
                "resnet34",
                "resnet34",
            ],
            "n_last_second_logit": [64 for _ in range(14)],
            "multi_head_instruction": [
                {"Thickness": 1} for _ in range(14)
            ],  # {"Thickness": 9, "Lift-off": 6}, {"Thickness": 4}
            "dropout_rate": [None for _ in range(14)],
            "resnet_v2": [
                None,
                None,
                None,
                False,
                True,
                False,
                True,
                None,
                None,
                None,
                False,
                True,
                False,
                True,
            ],
            "simpleCNN_list": [
                [128, 128, 128],
                None,
                [512, 826, 1024],
                None,
                None,
                None,
                None,
                [128, 128, 128],
                None,
                [512, 826, 1024],
                None,
                None,
                None,
                None,
            ],  # [128, 128, 128], [512, 826, 1024]
            "simpleCNN_kernel_shapes": [[3, 3, 3] for _ in range(14)],
            "simpleCNN_stride": [[2, 2, 2] for _ in range(14)],
            "mlp_list": [
                None,
                [1024, 1164, 2048],
                None,
                None,
                None,
                None,
                None,
                None,
                [1024, 1164, 2048],
                None,
                None,
                None,
                None,
                None,
            ],
            # -----------Dataset setting-----------
            "dataset_name": [
                "Q345_15112022",
                "Q345_15112022",
                "Q345_15112022",
                "Q345_15112022",
                "Q345_15112022",
                "Q345_15112022",
                "Q345_15112022",
                "aluminum",
                "aluminum",
                "aluminum",
                "aluminum",
                "aluminum",
                "aluminum",
                "aluminum",
            ],  # aluminum, Q345_data, circle_Q345_07102022, Q345_15112022
            "n_samples": [256 for _ in range(14)],
            "data_start_index": [0 for _ in range(14)],
            "split_ratio": [0.8 for _ in range(14)],
            "batch_size": [
                64,
                64,
                32,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
            ],  # Smaller better i.e 8, but training is slow.
            "z_norm_flag": [True for _ in range(14)],
            # -----------optimizer setting-----------
            "optimizer": ["adam" for _ in range(14)],
            "lr": [0.001 for _ in range(14)],
            "lr_schedule_flag": [True for _ in range(14)],
            # -----------Training setting-----------
            "epoch": [201 for _ in range(14)],
            # None, 0.0001
            "weight_decay": [None for _ in range(14)],
            # huber_loss, l2_loss, softmax_cross_entropy
            "loss_name": ["l2_loss" for _ in range(14)],
            # regression, classification
            "problem": ["regression" for _ in range(14)],
        }
    DATA_SIZE = (hparams["n_samples"][0], 1, 1)

    # Define PRNGKey.
    rng = jax.random.PRNGKey(666)

    if training_flag:
        # Loop for hyperparameters.
        for experiment in range(hparams["number_of_experiment"]):
            # Check if the experiment has already run.
            already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
                {
                    k: hparams[k][experiment]
                    for k in hparams.keys()
                    if k != "number_of_experiment"
                }
            )
            # If already ran, skip this experiment.
            if already_ran_flag:
                print(f"Experiment is skiped.")
                continue
            # Run experiment.
            with mlflow.start_run(
                run_id=previous_run_id,
                run_name=hparams["run_name"][experiment],
                description=hparams["description"][experiment],
            ) as active_run:
                # log hyper parameters.
                utils.mf_loghyperparams(hparams, experiment)

                # Get mlflow artifact saving path.
                mlflow_artifact_path = unquote(
                    urlparse(active_run.info.artifact_uri).path
                )
                if "t50851tm" in mlflow_artifact_path:
                    mlflow_artifact_path = os.path.relpath(
                        mlflow_artifact_path, "/net/scratch2/t50851tm/momaml_jax"
                    )

                # Load dataset.
                data_saved_path = join(data_root, "formatted_v2")
                dataset_name = hparams["dataset_name"][experiment]
                json_path = join(data_saved_path, dataset_name + ".json")
                data_dict = dataset_v2.load_json(json_path)
                train_dataset, val_dataset, test_dataset = dataset_v2.dataPipeline(
                    json_path,
                    splitRate=hparams["split_ratio"][experiment],
                    batchSize=hparams["batch_size"][experiment],
                    start_index=hparams["data_start_index"][experiment],
                    n_samples=hparams["n_samples"][experiment],
                    z_norm_flag=hparams["z_norm_flag"][experiment],
                )
                train_dataset_dict = visualization.convert_tfpipeline_to_dict(
                    train_dataset
                )

                # Define _forward and optimizer.
                (
                    _forward,
                    optimizer,
                    optimizerSchedule,
                    summary_message,
                ) = define_forward_and_optimizer(
                    hparams, experiment, dataset_name, DATA_SIZE
                )

                # Log model architecture.
                mlflow.log_text(
                    summary_message, join("summary", "model_architecture.txt")
                )

                # Transform forward-pass into pure functions.
                forward = hk.without_apply_rng(hk.transform_with_state(_forward))

                # Define training loss function.
                loss_fn = define_loss_fn(
                    forward,
                    is_training=True,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define train_step.
                train_step = define_train_step(loss_fn, optimizer, optimizerSchedule)

                # Define train.
                train = define_train(train_step, hparams, experiment)

                # Define test loss function.
                loss_fn_test = define_loss_fn(
                    forward,
                    is_training=False,
                    optax_loss=utils.lossSelector(hparams["loss_name"][experiment]),
                    hparams=hparams,
                    experiment=experiment,
                )

                # Define test.
                test = define_test(loss_fn_test, hparams, experiment)

                # Initialize train_exp_state.
                train_exp_state = initialize_train_exp_state(
                    DATA_SIZE, forward, optimizer, mlflow_artifact_path
                )

                # Check if restore checkpoint is needed.
                if starting_epoch != 0 and starting_epoch is not None:
                    train_exp_state = restore_exp_state(
                        starting_epoch, mlflow_artifact_path
                    )
                    if "best_val_loss" in active_run.data.metrics.keys():
                        best_val_loss = active_run.data.metrics["best_val_loss"]
                    else:
                        best_val_loss = 99999.9
                    print("Restored from Epoch", starting_epoch)
                else:
                    best_val_loss = 99999.9

                # Training loop.
                for epoch in range(starting_epoch, hparams["epoch"][experiment]):
                    start_time = time.time()
                    force_log_flag = False

                    # Update trainState.
                    train_exp_state, train_result = train(
                        train_exp_state, train_dataset
                    )
                    val_result = test(train_exp_state, val_dataset)
                    test_result = test(train_exp_state, test_dataset)

                    # Save model if train query loss is lower.
                    if (val_result["loss"] < best_val_loss) or (
                        epoch == hparams["epoch"][experiment] - 1
                    ):
                        save_exp_state(train_exp_state, epoch, mlflow_artifact_path)
                        force_log_flag = True
                    best_val_loss = min(val_result["loss"], best_val_loss)

                    if (
                        ((epoch % LOG_EPOCH == 0) and (epoch != 0))
                        or (epoch == hparams["epoch"][experiment] - 1)
                        or force_log_flag
                    ):
                        if "classification" in hparams["problem"][experiment]:
                            plot_confusion_matrix(
                                train_result,
                                val_result,
                                test_result,
                                train_dataset_dict,
                                hparams,
                                experiment,
                                epoch,
                                mlflow_artifact_path,
                            )
                        elif "regression" in hparams["problem"][experiment]:
                            mean_err_dict = plot_residual_fig2(
                                train_result,
                                val_result,
                                test_result,
                                hparams,
                                experiment,
                                epoch,
                                mlflow_artifact_path,
                            )
                            jsonSavePath = join(mlflow_artifact_path, "error_summary")
                            if not exists(jsonSavePath):
                                os.makedirs(jsonSavePath)
                            with open(
                                join(
                                    jsonSavePath,
                                    "error_summary_epoch" + str(epoch) + ".json",
                                ),
                                "w",
                            ) as f:
                                json.dump(mean_err_dict, f)

                        jsonSavePath = join(mlflow_artifact_path, "result")
                        if not exists(jsonSavePath):
                            os.makedirs(jsonSavePath)
                        for name, results in zip(
                            [
                                "trainResult_" + "epoch" + str(epoch) + ".json",
                                "validationResult_" + "epoch" + str(epoch) + ".json",
                                "testResult_" + "epoch" + str(epoch) + ".json",
                            ],
                            [train_result, val_result, test_result],
                        ):
                            with open(join(jsonSavePath, name), "w") as f:
                                json.dump(results, f)

                        train_ralative_error_dict = summary_result(
                            data_dict, train_result
                        )
                        val_ralative_error_dict = summary_result(data_dict, val_result)
                        test_ralative_error_dict = summary_result(
                            data_dict, test_result
                        )
                        ralative_error_summary_path = join(
                            mlflow_artifact_path, "ralative_error"
                        )
                        if not exists(ralative_error_summary_path):
                            os.makedirs(ralative_error_summary_path)

                        # Log train result.
                        mlflow.log_metric(
                            "train_loss", train_result["loss"], step=epoch
                        )
                        mlflow.log_metric(
                            "train_abs_error",
                            train_result["mean_abs_error"],
                            step=epoch,
                        )
                        mlflow.log_metric(
                            "train_ralative_error",
                            train_result["mean_ralative_error"],
                            step=epoch,
                        )
                        with open(
                            join(
                                ralative_error_summary_path,
                                "Epoch" + str(epoch) + "train_ralative_error.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(train_ralative_error_dict, f)

                        # Log validation result.
                        mlflow.log_metric(
                            "validation_loss", val_result["loss"], step=epoch
                        )
                        mlflow.log_metric(
                            "validation_abs_error",
                            val_result["mean_abs_error"],
                            step=epoch,
                        )
                        mlflow.log_metric(
                            "validation_ralative_error",
                            val_result["mean_ralative_error"],
                            step=epoch,
                        )
                        with open(
                            join(
                                ralative_error_summary_path,
                                "Epoch" + str(epoch) + "val_ralative_error.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(val_ralative_error_dict, f)

                        # Log test result.
                        mlflow.log_metric("test_loss", test_result["loss"], step=epoch)
                        mlflow.log_metric(
                            "test_abs_error",
                            test_result["mean_abs_error"],
                            step=epoch,
                        )
                        mlflow.log_metric(
                            "test_ralative_error",
                            test_result["mean_ralative_error"],
                            step=epoch,
                        )
                        with open(
                            join(
                                ralative_error_summary_path,
                                "Epoch" + str(epoch) + "test_ralative_error.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(test_ralative_error_dict, f)

                        # Log other metrics.
                        mlflow.log_metric("Epoch", epoch, step=epoch)
                        mlflow.log_metric(
                            "learning_rate", train_result["lr"][-1], step=epoch
                        )
                        mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                        mlflow.log_metric(
                            "grads_norm",
                            jnp.mean(jnp.array(train_result["grads_norm"])),
                            step=epoch,
                        )

                        print_message = f"Run id: {active_run.info.run_id} \n \
                                            Epoch: {epoch} \n \
                                            one epoch time: {time.time() - start_time} s \n \
                                            training loss: {train_result['loss']} \n \
                                            train_abs_error: {train_result['mean_abs_error']} \n \
                                            train_ralative_error: {train_result['mean_ralative_error']} \n \
                                            validation loss: {val_result['loss']} \n \
                                            validation_abs_error: {val_result['mean_abs_error']} \n \
                                            validation_ralative_error: {val_result['mean_ralative_error']} \n \
                                            test loss: {test_result['loss']} \n \
                                            test_abs_error: {test_result['mean_abs_error']} \n \
                                            test_ralative_error: {test_result['mean_ralative_error']} \n"
                        mlflow.log_text(
                            print_message, f"Message/Training_Epoch{epoch}.txt"
                        )
                        print(print_message)

            if convert_flag:
                # Get provious run ID.
                already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
                    {
                        k: hparams[k][experiment]
                        for k in hparams.keys()
                        if k != "number_of_experiment"
                    }
                )
                with mlflow.start_run(
                    run_id=previous_run_id, run_name=hparams["run_name"][experiment]
                ) as active_run:
                    mlflow_artifact_path = unquote(
                        urlparse(active_run.info.artifact_uri).path
                    )
                    # Restore train state.
                    train_exp_state = restore_exp_state(
                        starting_epoch, mlflow_artifact_path
                    )
                    # Define forword.
                    (
                        _forward,
                        optimizer,
                        optimizerSchedule,
                        summary_message,
                    ) = define_forward_and_optimizer(
                        hparams,
                        experiment,
                        hparams["dataset_name"][experiment],
                        DATA_SIZE,
                    )
                    # Transform forward-pass into pure functions.
                    forward = hk.without_apply_rng(hk.transform_with_state(_forward))

                    convert_to_tflite(
                        hparams,
                        experiment,
                        forward,
                        train_exp_state.diff["params"],
                        train_exp_state.non_diff["state"],
                        DATA_SIZE,
                        tflite_save_path=join(
                            mlflow_artifact_path,
                            "tflite_epoch" + str(starting_epoch),
                        ),
                    )

    if not training_flag:
        # Loop for hyperparameters.
        for experiment in range(hparams["number_of_experiment"]):
            # Get provious run ID.
            already_ran_flag, previous_run_id, starting_epoch = utils._already_ran(
                {
                    k: hparams[k][experiment]
                    for k in hparams.keys()
                    if k != "number_of_experiment"
                }
            )
            with mlflow.start_run(
                run_id=previous_run_id, run_name=hparams["run_name"][experiment]
            ) as active_run:
                mlflow_artifact_path = unquote(
                    urlparse(active_run.info.artifact_uri).path
                )
                # Restore train state.
                train_exp_state = restore_exp_state(
                    starting_epoch, mlflow_artifact_path
                )
                # Define forword.
                (
                    _forward,
                    optimizer,
                    optimizerSchedule,
                    summary_message,
                ) = define_forward_and_optimizer(
                    hparams,
                    experiment,
                    hparams["dataset_name"][experiment],
                    DATA_SIZE,
                )
                # Transform forward-pass into pure functions.
                forward = hk.without_apply_rng(hk.transform_with_state(_forward))

                convert_to_tflite(
                    hparams,
                    experiment,
                    forward,
                    train_exp_state.diff["params"],
                    train_exp_state.non_diff["state"],
                    DATA_SIZE,
                    tflite_save_path=join(
                        mlflow_artifact_path,
                        "tflite_epoch" + str(starting_epoch),
                    ),
                )


if __name__ == "__main__":
    # Ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type="GPU")
    DEVICE = "PECRuns"
    DATA_ROOT = join("datasets", "PEC")

    # Set random seed.
    np.random.seed(666)
    tf.random.set_seed(667)

    main(
        training_flag=False,
        DEVICE=DEVICE,
        data_root=DATA_ROOT,
        convert_flag=True,
    )
