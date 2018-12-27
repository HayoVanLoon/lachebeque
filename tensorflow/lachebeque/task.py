import argparse
import json
import os

from . import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_data_path',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--eval_data_path',
        help='GCS or local path to evaluation data',
        required=True
    )
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--learning_rate',
        help='Initial learning rate for training',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--train_steps',
        help="""\
      Steps to run the training job for. A step is one batch-size,\
      """,
        type=int,
        default=0
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    model_names = [name.replace('_model', '') \
                   for name in dir(model) \
                   if name.endswith('_model')]
    parser.add_argument(
        '--model',
        help='Type of model. Supported types are {}'.format(model_names),
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Minimum number of training steps between evaluations',
        default=60,
        type=int
    )

    parser.add_argument(    # TODO: truncate past max?
        '--max_sequence_length',
        help="""\
      Maximum length (in words) of a joke. If jokes exceed this length, an error will be thrown.
      """,
        type=int,
        default=120
    )
    parser.add_argument(
        '--debug',
        help="""\
      Show debug messages
      """,
        type=bool,
        default=False
    )
    parser.add_argument(
        '--score_type',
        help="""\
          Score to use [score | norm_score | norm_log_score]
          """,
        type=str,
        default='norm_log_score'
    )

    # Hyperparameters
    parser.add_argument(
        '--cell_size',
        help="""\
      GRU cell size
      """,
        type=int,
        default=10
    )
    parser.add_argument(
        '--hidden_layer_size',
        help="""\
      GRU cell size
      """,
        type=int,
        default=10
    )

    args = parser.parse_args()
    hparams = args.__dict__

    # unused args provided by service
    hparams.pop('job_dir', None)
    hparams.pop('job-dir', None)

    output_dir = hparams.pop('output_dir')

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # calculate train_steps if not provided
    if hparams['train_steps'] < 1:
        # 1,000 steps at batch_size of 100
        hparams['train_steps'] = (1000 * 100) // hparams['train_batch_size']
        print("Training for {} steps".format(hparams['train_steps']))

    model.init(hparams)

    # Run the training job
    model.train_and_evaluate(output_dir, hparams)
