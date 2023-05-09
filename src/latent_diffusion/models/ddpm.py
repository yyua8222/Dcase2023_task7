import pdb
import sys
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
import numpy as np

from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from latent_diffusion.util import (
    log_txt_as_img,
    exists,
    default,
    ismap,
    isimage,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from latent_diffusion.modules.ema import LitEma
from latent_diffusion.modules.distributions.distributions import (
    normal_kl,
    DiagonalGaussianDistribution,
)
from latent_encoder.autoencoder import (
    VQModelInterface,
    IdentityFirstStage,
    AutoencoderKL,
)
from latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
from latent_diffusion.models.ddim import DDIMSampler
from latent_diffusion.models.plms import PLMSSampler
import soundfile as sf
import os
import torch.nn.functional as F
import torchaudio


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.state = None
        # print(
        #     f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        # )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_project = None
        self.logger_version = None
        self.label_indices_total = None
        # To avoid the system cannot find metric value for checkpoint
        self.metrics_buffer = {
            "val/kullback_leibler_divergence_sigmoid": 15.0,
            "val/kullback_leibler_divergence_softmax": 10.0,
            "val/psnr": 0.0,
            "val/ssim": 0.0,
            "val/inception_score_mean": 1.0,
            "val/inception_score_std": 0.0,
            "val/kernel_inception_distance_mean": 0.0,
            "val/kernel_inception_distance_std": 0.0,
            "val/frechet_inception_distance": 133.0,
            "val/frechet_audio_distance": 32.0,
        }
        self.initial_learning_rate = None
        self.test_data_subset_path = None
        
    def get_log_dir(self):
        if (
            self.logger_save_dir is None
            and self.logger_project is None
            and self.logger_version is None
        ):
            return os.path.join(
                self.logger.save_dir, self.logger._project, self.logger.version
            )
        else:
            return os.path.join(
                self.logger_save_dir, self.logger_project, self.logger_version
            )

    def set_log_dir(self, save_dir, project, version):
        self.logger_save_dir = save_dir
        self.logger_project = project
        self.logger_version = version

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            #     print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                #     print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        # fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
        fbank, log_magnitudes_stft, label_indices, fname, waveform, text = batch
        ret = {}

        ret["fbank"] = (
            fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        )
        ret["stft"] = log_magnitudes_stft.to(
            memory_format=torch.contiguous_format
        ).float()
        # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
        ret["text"] = list(text)
        ret["fname"] = fname

        return ret[k]

    def get_validation_folder_name(self):
        return "val_%s" % (self.global_step)


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        batchsize=None,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        has_embed=False,
        *args,
        **kwargs,
    ):
        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.evaluation_params = evaluation_params
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        # self.transform = torchaudio.transforms.Resample(16000, 22050)
        self.transform = None
        if has_embed:
            self.embed_v = nn.Linear(513,100)

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True



    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model
        self.cond_stage_model = self.cond_stage_model.to(self.device)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                if len(c) == 1:
                    c = self.cond_stage_model([c[0], c[0]])
                    c = c[0:1]
                else:
                    c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c


    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=False,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        x = super().get_input(batch, k)

        if bs is not None:
            x = x[:bs]

        x = x.to(self.device)

        # if return_first_stage_encode:
        #     encoder_posterior = self.encode_first_stage(x)
        #     z = self.get_first_stage_encoding(encoder_posterior).detach()
        # else:
        #     z = None

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    # [bs, 1, 527]
                    xc = super().get_input(batch, cond_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc

            if bs is not None:
                c = c[:bs]

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = c
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i],
                            force_not_quantize=predict_cids or force_not_quantize,
                        )
                        for i in range(z.shape[-1])
                    ]
                else:

                    output_list = [
                        self.first_stage_model.decode(z[:, :, :, :, i])
                        for i in range(z.shape[-1])
                    ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(
                        z, force_not_quantize=predict_cids or force_not_quantize
                    )
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(
                    z, force_not_quantize=predict_cids or force_not_quantize
                )
            else:
                return self.first_stage_model.decode(z)


    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def save_waveform(self, waveform, savepath, name="outwav",saved = True):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            pre_sr = self.cond_stage_model.sampling_rate
            # if pre_sr == 22050:

            if pre_sr == 16000:
                print("upsamplying to 22050hz")
                waveform = torch.Tensor(waveform)
                if self.transform:
                    self.transform = torchaudio.transforms.Resample(16000, 22050)
                    waveform = self.transform(waveform)
                else:
                    self.transform = torchaudio.transforms.Resample(16000,22050)
                    waveform = self.transform(waveform)
                waveform = waveform.cpu().detach().numpy()

                pre_sr =22050

            writen_wav = waveform[i,0]
            if pre_sr == 22050:
                wav_length = 22050*4
                writen_length = writen_wav.shape[0]

                if writen_length <wav_length:
                    zeros = np.zeros(wav_length-writen_length)
                    writen_wav = np.concatenate((writen_wav,zeros))
                else:
                    writen_wav = writen_wav[:wav_length]


                # print(f"the wavefrom is {writen_wav.shape} and the writen is {waveform[i,0].shape}")


            if saved:
                sf.write(path, writen_wav, samplerate=22050)
            return writen_wav

            # print("the learning rate is ",self.cond_stage_model.sampling_rate)


    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    x, ks, stride, df=df
                )
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                output_list = [
                    self.first_stage_model.encode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)


    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            # if self.shorten_cond_schedule:  # TODO: drop this option
            #     tc = self.cond_ids[t].to(self.device)
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        loss = self.p_losses(x, c, t, *args, **kwargs)
        return loss


    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"

            cond = {key: cond}
        padded = False
        # print(f"!!!!!!!!!!!! the x_noisy shape is{x_noisy.shape} and t is {t.shape}")
        if x_noisy.shape[2] == 215:
            # print("start padding")
            x_noisy=F.pad(input=x_noisy,pad=(2,2,0,1),mode='constant',value=0)
            # print(f"the shape after padding is {x_noisy.shape}")
            padded = True

        if x_noisy.shape[2] == 100:
            # print("start padding")
            x_noisy=F.pad(input=x_noisy,pad=(0,0,2,2),mode='constant',value=0)
            # print(f"the shape after padding is {x_noisy.shape}")
            padded = True

        if x_noisy.shape[2] == 86:
            if x_noisy.shape[3] == 16:
                x_noisy = F.pad(input=x_noisy, pad=(0, 0, 1, 1), mode='constant', value=0)
            # print("start padding")
            else:
                x_noisy=F.pad(input=x_noisy,pad=(2,2,1,1),mode='constant',value=0)
            # print(f"the shape after padding is {x_noisy.shape}")
            padded = True

        x_recon = self.model(x_noisy, t, **cond)

        if padded:
            # print("start re_pading")
            if x_noisy.shape[2] == 216:
                x_recon = x_recon[:,:,:-1,2:-2]
            if x_noisy.shape[2] == 104:
                x_recon = x_recon[:, :, 2:-2, :]
            if x_noisy.shape[2] == 88:
                if x_noisy.shape[3] ==16:
                    x_recon = x_recon[:, :, 1:-1,:]
                else:
                    x_recon = x_recon[:,:,1:-1,2:-2]
            # print(f"the reconed after repad is {x_recon.shape}")


        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        x_T=None,
        **kwargs,
    ):

        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)


        intermediate = None
        # print("Use ddim sampler")

        ddim_sampler = DDIMSampler(self)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            batch_size,
            shape,
            cond,
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            mask=mask,
            x_T=x_T,
            **kwargs,
        )

        return samples, intermediate




    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        limit=0.21,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        use_transfer=False,
        clap = None,
        target = -1,
        change_limit = 0,
        saved = True,
        wave_save_path = "results3",
        **kwargs,
    ):
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        # waveform_save_path = os.path.join(self.get_log_dir(), name)
        waveform_save_path = os.path.join(wave_save_path,name)
        os.makedirs(waveform_save_path, exist_ok=True)

        if (
            "audiocaps" in waveform_save_path
            and len(os.listdir(waveform_save_path)) >= 964
        ):
            print("The evaluation has already been done at %s" % waveform_save_path)
            return waveform_save_path

        # with self.ema_scope("Plotting"):
        for batch in batchs:
            c = self.get_input(
                batch,
                self.first_stage_key,
                return_first_stage_outputs=False,
                force_c_encode=True,
                return_original_cond=False,
                bs=None,
            )
            text = super().get_input(batch, "text")

            # print("the z shape is",z.shape[0])

            batch_size = 1 * n_gen
            c = torch.cat([c] * n_gen, dim=0)
            text = text * n_gen

            if use_transfer:
                x_T = torch.cat([z]*n_gen,dim=0)

            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = (
                    self.cond_stage_model.get_unconditional_condition(batch_size)
                )

            fnames = list(super().get_input(batch, "fname"))

            samples, _ = self.sample_log(
                cond=c,
                batch_size=batch_size,
                x_T=x_T,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                use_plms=use_plms,
            )

            mel = self.decode_first_stage(samples)

            waveform = self.mel_spectrogram_to_waveform(
                mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
            )

            # print(f"the waveform shape is {waveform.shape} and the text is {text}")
            if target>=0:
                limit = self.embed_v.weight[target][-1].cpu().detach().numpy()+0
                if change_limit>0:
                    limit = limit-0.05*change_limit
                tarwav = torch.cat([self.embed_v.weight[target][:-1].reshape(1,-1)]*n_gen,dim=0)
                waveforms = torch.FloatTensor(waveform).squeeze(1)
                try:
                    similarity = self.cond_stage_model.emb_similarity(waveforms, tarwav.reshape(n_gen,1,-1))
                except:
                    pdb.set_trace()

            else:
                if n_gen==1:
                    txt = text+text
                    wav = torch.FloatTensor(waveform).squeeze(1)
                    waveforms = torch.cat([wav, wav])
                    similarity = self.cond_stage_model.cos_similarity(waveforms, txt)
                    while similarity[0]!=similarity[1]:
                        # print("not equal!")
                        similarity = self.cond_stage_model.cos_similarity(waveforms, txt)
                else:
                    similarity = self.cond_stage_model.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )

                    new_similarity = self.cond_stage_model.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )
                    compare = similarity==new_similarity

                    while False in compare:
                        similarity = new_similarity
                        new_similarity = self.cond_stage_model.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                        )
                        # print("not equal! ")
                        compare = similarity==new_similarity

                    if clap:
                        print("calculate clap score")

            best_index = []
            # for i in range(z.shape[0]):
            candidates = similarity
            max_index = torch.argmax(candidates).item()
            best_index.append(max_index)
                # print(f"from candidates {candidates} choose number {max_index}")

            waveform = waveform[best_index]
            best_score = similarity[best_index]
            # print(f"Similarity between generated audio {similarity} and text{text}")
            # print(f"Choose the following indexes:{best_index} and score {best_score}")
            # print(f"Waveform save path: {waveform_save_path}/{fnames}")

            cur_text = text[0]

            if best_score < limit:
                # print(f"score is too low for score {limit}")
                return False,torch.FloatTensor(waveform).squeeze(1),False
            if best_score > 0.99:
                # print(f"score is too high as  {best_score}"),False

                return False,torch.FloatTensor(waveform).squeeze(1),False

            result_wav = self.save_waveform(waveform, waveform_save_path, name=fnames,saved=saved)
        return best_score,torch.FloatTensor(waveform).squeeze(1),result_wav


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [
            None,
            "concat",
            "crossattn",
            "hybrid",
            "adm",
            "film",
        ]

    def forward(
        self, x, t, c_concat: list = None, c_crossattn: list = None, c_film: list = None
    ):
        x = x.contiguous()
        t = t.contiguous()

        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif (
            self.conditioning_key == "film"
        ):  # The condition is assumed to be a global token, which wil pass through a linear layer and added with the time embedding for the FILM
            cc = c_film[0].squeeze(1)  # only has one token
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
