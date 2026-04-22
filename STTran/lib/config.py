from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-6
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', '--mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', '--save_path', default='data/', type=str)
        parser.add_argument('-model_path', '--model_path', default=None, type=str)
        parser.add_argument('-data_path', '--data_path', default='/data/scene_understanding/action_genome/', type=str)
        parser.add_argument('-datasize', '--datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', '--ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', '--optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', '--lr', dest='lr', help='learning rate', default=1e-6, type=float)
        parser.add_argument('-nepoch', '--nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('-enc_layer', '--enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', '--dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('-bce_loss', '--bce_loss', action='store_true')
        parser.add_argument('--backbone', dest='backbone', help='resnet101 or vitdet', default='resnet101', type=str)
        parser.add_argument(
            '--det_threshold',
            dest='det_threshold',
            help='score threshold for detector class filtering in sgdet',
            default=0.1,
            type=float,
        )
        parser.add_argument('--max_train_steps', dest='max_train_steps', default=None, type=int)
        parser.add_argument('--max_test_steps', dest='max_test_steps', default=None, type=int)
        return parser
