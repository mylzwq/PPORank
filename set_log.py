from __future__ import print_function
import logging
import datetime
from arguments import get_args
import os


def initialize_logger_name(args):

    if args.analysis == 'KEEPK':
        kr = args.keepk_ratio
        args.k = args.keepk
    else:
        kr = 0

    if args.full is True or args.full == 'True':
        rank_str = "FULL"
    else:
        rank_str = args.k

    if args.analysis == 'KEEPK':
        keepk = args.keepk
    elif args.analysis == 'FULL':
        keepk = ""
    else:
        keepk = args.scenario

    miss_rate = args.miss_rate if args.analysis == "sparse" else ""
    scale = args.scale if args.normalize_y else "raw"

    if args.algo == 'ppo':
        # NAME = "_".join(("{}".format(args.algo),
        #                  "{}".format(args.Data),
        #                  "{}".format(args.analysis),
        #                  "{}Dim".format(args.f),
        #                  "{}".format(keepk),
        #                  "{}scale".format(args.normalize_y),
        #                  "VF{}".format(args.value_loss_coef),
        #                  "B{}".format(args.num_processes),
        #                  "gamma{}".format(args.gamma),
        #                  "lambda{}".format(args.gae_lambda),
        #                  "{}Dlayers".format(args.nlayers_deep),
        #                  "{}Clayers".format(args.nlayers_cross),
        #                  "{}".format(args.fold),
        #                  "ratio{}".format(kr)
        #                  ))
        NAME = "_".join(("{}".format(args.algo),
                         "{}".format(args.Data),
                         "{}".format(args.analysis),
                         "f{}".format(args.f),
                         "{}".format(keepk),
                         "S{}".format(scale),  # ('.{}'.format(fold2)) if fold2 != -1 else ''
                         "B{}".format(args.num_processes),
                         "D{}".format(args.nlayers_deep),
                         "C{}".format(args.nlayers_cross),
                         "{}".format(args.fold)
                         ))
    elif args.algo in ['dnn', 'pg']:
        NAME = "_".join(("{}".format(args.algo),
                         "{}".format(args.Data),
                         "{}".format(args.analysis),
                         "{}Dim".format(args.f),
                         "{}".format(keepk),
                         "{}scale".format(scale),
                         "VF{}".format(args.value_loss_coef),
                         "B{}".format(args.num_processes),
                         "gam{}".format(args.gamma),
                         "la{}".format(args.gae_lambda),
                         "{}D".format(args.nlayers_deep),
                         "{}C".format(args.nlayers_cross),
                         "{}".format(args.fold)
                         ))
    if not args.pretrain:
        NAME = NAME + "noP"
    if miss_rate:
        NAME += "miss{}".format(miss_rate)

    if args.augment:
        NAME += "_SD{}".format(args.sigma)+"_MU{}".format(args.mu)

    return NAME


def set_logging(NAME):
    PROCESS_START_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    formatter = logging.Formatter(
        '%(asctime)s %(name)12s [%(levelname)s]\t%(message)s')
    ch = logging.StreamHandler()  # TqdmLoggingHandler()  #
    ch.setFormatter(formatter)
    if not os.path.exists(os.path.join('./logs', "{}".format(NAME))):
        os.makedirs(os.path.join('./logs', "{}".format(NAME)))
    fh = logging.FileHandler(os.path.join(
        './logs', "{}" "/{}.log".format(NAME, PROCESS_START_TIME)))
    fh.setFormatter(formatter)

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger(__name__)
    logger.info("Model Name is {}".format(NAME))

    return logger
