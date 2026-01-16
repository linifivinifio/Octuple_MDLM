import os

class HparamsBase(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    def apply_parser_values(self, parser):
        # NOTE default args will be overwritten by any default parser args
        parser = parser.__dict__
        for arg in parser:
            if parser[arg] is not None:
                self[arg] = parser[arg]


class HparamsAbsorbing(HparamsBase):

    def __init__(self, parser):
        super().__init__()

        self.sampler = "absorbing"
        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.sample_schedule = "random"
        self.attn_pdrop = 0.2
        self.embd_pdrop = 0.2
        self.resid_pdrop = 0.2
        self.temp = 1.0
        self.steps_per_eval = 20
        self.steps_per_checkpoint = 50
        self.steps_per_log = 10
        self.steps_per_update_ema = 10
        self.steps_per_sample = 50
        self.load_step = 0
        self.sampling_batch_size = 24
        self.bert_n_emb = 512
        self.bert_n_head = 8
        self.bert_n_layers = 24
        self.lr = 5e-4
        self.warmup_iters = 10000
        self.validation_set_size = 0.05
        self.augment = False

        self.apply_parser_values(parser)

        self.NOTES = self.bars * 16
        self.total_steps = self.NOTES
        self.sample_steps = self.NOTES
        self.block_size = self.NOTES
        self.tracks = self.tracks
        if self.log_base_dir:
            self.log_dir = os.path.join(self.log_base_dir, f'log_{self.model}_{self.tracks}_{self.NOTES}')
        else:
            self.log_dir = f'log_{self.model}_{self.tracks}_{self.NOTES}'
        if not self.load_dir:
            self.load_dir = self.log_dir
        print("DEEBUG: Tracks: ", self.tracks)
        self.codebook_size = (128, ) if self.tracks == 'melody' else (128, 128, 128)
        self.latent_shape = (self.NOTES, len(self.codebook_size))
        self.load_optim = self.load_step != 0


class HparamsAbsorbingConv(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.bert_n_emb = 512
        self.conv_layers = 1
        self.conv_len = 4


class HparamsHierarchTransformer(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.sub_seq_len = 32
        self.bert_n_emb = 512
        self.upper_bert_n_emb = 512
        self.augment = False


class HparamsUTransformer(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.layers_per_level = 2
        self.bert_n_emb = 512
        self.conv_width = 4
        self.augment = False


class HparamsOctuple(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        
        # time, tempo, bar, position, instrument/program, pitch, durration, velocity
        self.codebook_size = (260, 132, 133, 132, 132, 36, 258, 53)
        self.latent_shape = (self.NOTES, 8)

