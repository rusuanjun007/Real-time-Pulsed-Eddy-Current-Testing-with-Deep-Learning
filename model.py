import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax_helper.haiku_model.cnn import CNN2D as hk_CNN2D
from jax_helper.haiku_model.mlp import MLP as hk_MLP
from jax_helper.haiku_model.multihead import MultiHead as hk_MultiHead
from jax_helper.haiku_model.resnet import ResNet18 as hk_ResNet18
from jax_helper.haiku_model.resnet import ResNet34 as hk_ResNet34
from jax_helper.tf_model.cnn import CNN2DOps as tf_CNN2DOps
from jax_helper.tf_model.mlp import MLPOps as tf_MLPOps
from jax_helper.tf_model.multihead import MultiHeadOps as tf_MultiHeadOps
from jax_helper.tf_model.resnet import CONFIGS as tf_ResNet_CONFIGS
from jax_helper.tf_model.resnet import ResNetOps as tf_ResNetOps


def hk_PEC_model(hparams: dict, experiment: int):
    model_name = hparams["model_name"][experiment]
    n_last_second_logit = hparams["n_last_second_logit"][experiment]

    def _forward(x: np.ndarray, is_training: bool) -> jnp.ndarray:
        # Define forward-pass.
        if "resnet18" == model_name:
            module = hk_ResNet18(
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
                name=model_name,
                is_timeseries=True,
            )
            x = module(x, is_training)
        elif "resnet34" == model_name:
            module = hk_ResNet34(
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
                name=model_name,
                is_timeseries=True,
            )
            x = module(x, is_training)
        elif "simpleCNN" == model_name:
            module = hk_CNN2D(
                linear_outputs=[n_last_second_logit],
                cnn_channels=hparams["simpleCNN_list"][experiment],
                kernel_shapes=hparams["simpleCNN_kernel_shapes"][experiment],
                strides=hparams["simpleCNN_stride"][experiment],
                bn_decay_rate=0.9,
                activation_fn=jax.nn.relu,
                dropout_rate=hparams["dropout_rate"][experiment],
                feature_group_count=1,
                name=model_name,
                is_timeseries=True,
            )
            x = module(x, is_training)
        elif "mlp" == model_name:
            module = hk_MLP(
                linear_list=hparams["mlp_list"][experiment],
                output_size=n_last_second_logit,
                bn_decay_rate=0.9,
                activation_fn=jax.nn.relu,
                dropout_rate=None,
                name=model_name,
            )
            x = module(x, is_training)
        else:
            print(f"Model name {model_name} incorrect.")
            assert False

        x = hk_MultiHead(
            hparams["multi_head_instruction"][experiment],
            activation_fn=jax.nn.relu,
            name="multihead",
        )(x)

        return x

    return _forward


def tf_PEC_model(DATA_SIZE, hparams: dict, experiment: int):
    model_name = hparams["model_name"][experiment]
    n_last_second_logit = hparams["n_last_second_logit"][experiment]

    def _combined_model(input_shape: tuple) -> tf.Tensor:
        inputs = tf.keras.Input(shape=input_shape)

        # Define call stack.
        if "resnet18" == model_name:
            x = tf_ResNetOps(
                inputs,
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
                name=model_name,
                is_timeseries=True,
                **tf_ResNet_CONFIGS[18],
            )
        elif "resnet34" == model_name:
            x = tf_ResNetOps(
                inputs,
                num_classes=n_last_second_logit,
                resnet_v2=hparams["resnet_v2"][experiment],
                name=model_name,
                is_timeseries=True,
                **tf_ResNet_CONFIGS[34],
            )
        elif "simpleCNN" == model_name:
            x = tf_CNN2DOps(
                inputs,
                linear_outputs=[n_last_second_logit],
                cnn_channels=hparams["simpleCNN_list"][experiment],
                kernel_shapes=hparams["simpleCNN_kernel_shapes"][experiment],
                strides=hparams["simpleCNN_stride"][experiment],
                bn_decay_rate=0.9,
                activation_fn=tf.keras.activations.relu,
                dropout_rate=hparams["dropout_rate"][experiment],
                feature_group_count=1,
                name=model_name,
                is_timeseries=True,
            )
        elif "mlp" == model_name:
            x = tf_MLPOps(
                inputs,
                linear_list=hparams["mlp_list"][experiment],
                output_size=n_last_second_logit,
                bn_decay_rate=0.9,
                activation_fn=tf.keras.activations.relu,
                dropout_rate=hparams["dropout_rate"][experiment],
                name=model_name,
            )
        else:
            print(f"Model name {model_name} incorrect.")
            assert False

        x = tf_MultiHeadOps(
            x,
            hparams["multi_head_instruction"][experiment],
            activation_fn=tf.keras.activations.relu,
            name="multihead",
        )

        # Create model.
        model = tf.keras.Model(inputs, x, name=model_name)

        return model

    return _combined_model(DATA_SIZE)
