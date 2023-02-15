""""Model definitions."""
import tensorflow as tf
import babel from '@rollup/plugin-babel'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import { terser } from 'rollup-plugin-terser'

const output = {
  name: 'StimulusCheckAll',
    format: 'umd',
      sourcemap: true,
        globals: { stimulus: 'Stimulus' }
        }

        export default {
          input: 'src/index.js',
            output: [
                {
                      ...output,
                            file: 'dist/stimulus-check-all.umd.js'
                                },
                                    {
                                          ...output,
                                                file: 'dist/stimulus-check-all.umd.min.js',
                                                      plugins: [terser ()]
                                                          }
                                                            ],
                                                              external: ['stimulus'],
                                                                plugins: [
                                                                    babel({ babelHelpers: 'bundled' }),
                                                                        resolve({ resolveOnly: ['@github/check-all'] }),
                                                                            commonjs()

def shift_detection_model(
        enhancement_module,
        comparison_model,
        input_channels=64
):
    """Shift detection model.

    Parameters
    ----------
    enhancement_module: tf.keras.Model
        Model that enhances brain response data.
    comparison_model: tf.keras.Model
        Model that compares enhanced brain response data and predicts how
        much the second brain response is shifted relative to the first.
    input_channels: int
        Number of channels in the input data.


    Returns
    -------
    tf.keras.Model
        Model that detects the shift in time between two brain response
        segments, evoked by the same stimulus.

    Notes
    -----
    The model takes two inputs, each of shape (batch_size, time, channels).
    Brain
    """
    brain_response_1 = tf.keras.layers.Input((None, input_channels))
    brain_response_2 = tf.keras.layers.Input((None, input_channels))

    enhanced_response_1 = enhancement_module(brain_response_1)
    enhanced_response_2 = enhancement_module(brain_response_2)

    out = comparison_model([enhanced_response_1, enhanced_response_2])
    return tf.keras.models.Model(
        inputs=[brain_response_1, brain_response_2],
        outputs=[out],
        name="shift_detection_model",
    )


def multiview_cnn(filters=64, input_channels=64, output_channels=64):
    """Model based on the multi-view CNN of [1]_."

    Parameters
    ----------
    filters: int
        Number of filters to use for the parallel convolutional layers.
    input_channels: int
        Number of channels in the input data.
    output_channels: int
        Number of channels in the output data.


    Returns
    -------
    tf.keras.Model
        Model based on the multiview CNN architecture of [1]_. Can be used
        as a enhancement module in the shift detection model.

    References
    ----------
    .. [1] H. Su, S. Maji, E. Kalogerakis, and E. Learned-Miller,
       “Multi-View Convolutional Neural Networks for 3D Shape Recognition,”
       presented at the Proceedings of the IEEE International Conference on
       Computer Vision, 2015, pp. 945–953.

    """
    eeg = tf.keras.layers.Input((None, input_channels))

    x = tf.keras.layers.Dense(128)(eeg)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    all_times = []
    for kernel in [4, 8, 16, 24, 32, 40, 48, 56, 64]:
        a = tf.keras.layers.Conv1D(filters, kernel)(x)
        a = tf.keras.layers.LayerNormalization()(a)
        a = tf.keras.layers.LeakyReLU()(a)

        a = tf.keras.layers.ZeroPadding1D((0, kernel - 1))(a)

        a = tf.keras.layers.Dense(filters)(a)
        a = tf.keras.layers.LayerNormalization()(a)
        a = tf.keras.layers.LeakyReLU()(a)

        all_times += [a]

    x = tf.keras.layers.Concatenate()(all_times)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(output_channels)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    return tf.keras.models.Model(
        inputs=[eeg], outputs=[x], name="multiview_cnn"
    )


def simple_comparison_model(input_channels=64, nb_output_classes=13):
    """Simple comparison model.

    Parameters
    ----------
    input_channels: int
        Number of channels in the input data.
    nb_output_classes: int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Simple comparison model that can be used as a comparison model in the
        shift detection model.
    """
    enhanced_response_1 = tf.keras.layers.Input((None, input_channels))
    enhanced_response_2 = tf.keras.layers.Input((None, input_channels))

    cosine_scores = tf.keras.layers.Dot(1, normalize=True)(
        [enhanced_response_1, enhanced_response_2]
    )
    all_scores = tf.keras.layers.Flatten()(cosine_scores)
    out = tf.keras.layers.Dense(nb_output_classes, activation="softmax")(
        all_scores
    )
    return tf.keras.models.Model(
        inputs=[enhanced_response_1, enhanced_response_2],
        outputs=[out],
        name="simple_comparison_model",
    )


def linear_decoder(filters=1, integration_window=32):
    """A linear decoder.

    Returns
    -------
    tf.keras.Model
        A linear decoder. In the paper, this model is used to decode the speech
        envelope from EEG
    """
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(filters, integration_window),
        ],
        name="linear_decoder",
    )
