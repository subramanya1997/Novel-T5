import argparse

def add_cmdline_args_gen(argparser):
    argparser.add_argument_group("Huggingface models' .generate() method "
                                 "arguments")
    argparser.add_argument(
        '--sampling',
        action="store_true",
        default=False,
        help='Perform sampling'
    )
    argparser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size, if 1 then greedy search',
    )
    argparser.add_argument(
        '-Nbest',
        type=int,
        default=1,
        help='Number of return sequences',
    )
    argparser.add_argument(
        '--beam-length-penalty',
        type=float,
        default=0.65,
        help='Applies a length penalty. Set to 0 for no penalty.',
    )
    argparser.add_argument(
        #'--topk', type=int, default=10, help='K used in Top K sampling'
        '--topk', type=int, default=None, help='K used in Top K sampling'
    )
    argparser.add_argument(
        #'--topp', type=float, default=0.9, help='p used in nucleus sampling'
        '--topp', type=float, default=None, help='p used in nucleus sampling'
    )

    argparser.add_argument(
        #'--temp',type=float,default=1.0,help='temperature to add during
        # decoding',
        '--temp',type=float,default=None,help='temperature to add during '
                                            'decoding',
    )

    return argparser
