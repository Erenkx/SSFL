"""
This module provides a utility function for logging and saving model
size and training configuration.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

import os


def print_options(args, model):
    """
    Prints and saves the model size and training configuration.

    Args:
        args: Command-line arguments containing training configuration.
        model: The model instance being trained.
    """
    # Compute total number of trainable parameters in millions
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / 1e6

    message = (
        f'======== FL training of {args.model} ({num_params:2.1f}M) ========\n'
        '++++++++ Other training related parameters ++++++++\n'
    )

    # Log all training arguments
    for k, v in sorted(vars(args).items()):
        message += f'{str(k):>25}: {str(v):<30}\n'

    message += '++++++++ End of showing parameters ++++++++'

    print(message)

    # Save the log to disk
    log_file_path = os.path.join(args.output_dir, 'log_file.txt')
    with open(log_file_path, 'w') as fout:
        fout.write(message + '\n')
