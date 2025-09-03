"""
Utility for logging model size and training configuration.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

import os


def print_options(args, model):
    """
    Prints and saves the model size and training configuration.
    """
    # Compute total number of trainable parameters in millions
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) / 1e6

    message = (
        f'======== FL training of {args.model} with total model parameters: ' \
        f'{num_params:2.1f}M ========\n' \
        '++++++++ Other training related parameters ++++++++\n'
    )

    # Log all training arguments
    for k, v in sorted(vars(args).items()):
        message += f'{str(k):>25}: {str(v):<30}\n'

    message += '++++++++ End of showing parameters ++++++++'

    print(message)

    # Save the log to disk
    args.file_name = os.path.join(args.output_dir, 'log_file.txt')
    with open(args.file_name, 'w') as fout:
        fout.write(message + '\n')
