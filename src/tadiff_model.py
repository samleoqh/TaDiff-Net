import numpy as np
import torch
from src.net.tadiff_unet_arch import TaDiff_Net
# import wandb # logging metrics

from pytorch_lightning import LightningModule, Callback
from torch.optim import AdamW, SGD
from src.net.ssim import SSIM

from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from src.net.diffusion import GaussianDiffusion
import torch.nn.functional as F


from monai.losses.dice import DiceLoss, GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric

class Tadiff_model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = config
        self._model = TaDiff_Net(
            image_size=self.cfg.image_size, 
            in_channels=self.cfg.in_channels-1, 
            out_channels=self.cfg.out_channels,
            # num_intv_time=self.cfg.num_intv_time,
            model_channels=self.cfg.model_channels, 
            num_res_blocks=self.cfg.num_res_blocks, 
            channel_mult=self.cfg.channel_mult,
            attention_resolutions=self.cfg.attention_resolutions, 
            num_heads=self.cfg.num_heads, 
            )
        
        # if self.cfg.precision=='16':
        #     self._model.convert_to_fp16()
        
        self.diffusion = GaussianDiffusion(T=self.cfg.max_T, schedule=self.cfg.ddpm_schedule)#'linear')
        self.alphabar = np.cumprod(1-np.linspace(1e-4, 2e-2, self.cfg.max_T))
        
        # self.diffusion = LinearDiffusion(T=self.cfg.max_T)#'linear')
        self.best_val_loss = None
        self.best_val_epoch = 0
        self.val_step_outputs = [] # for callback
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        # self.dilation3 = torch.ones(1,1,3,3)
        self.dilation_filters = torch.ones(1,1,11,11) / 10.
        # self.dice = DiceLoss(include_background=False, sigmoid=True)
        self.dice = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
                             to_onehot_y=False, sigmoid=True, reduction="none")
        # self.dice = GeneralizedDiceFocalLoss( to_onehot_y=False, sigmoid=True, reduction="none")
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        # self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch") # per class dice

    def forward(self, x, timesteps, intv_t, treat_code, i_tg=None):
        return self._model(x, timesteps, intv_t,  treat_code, i_tg)
    
    def load_model(self, path=None, device='cuda:0'):
        # for loading old trained model without using pytorch-lightning
        if path is not None:
            self._model.load_state_dict(torch.load(path, map_location=device), strict=False)
        # self._model.load_state_dict(torch.load(path, map_location=device), strict=False)
        self._model.eval().to(device)
        print('Model Created!')

    def configure_optimizers(self):
        if self.cfg.opt == 'adamw':
            optimizer = AdamW(self.trainer.model.parameters(), 
                            lr=float(self.cfg.lr), 
                            weight_decay=self.cfg.weight_decay
                            )
        else:
            optimizer = SGD(self.trainer.model.parameters(), 
                            lr=float(self.cfg.lr), 
                            momentum = 0.9, 
                            nesterov = True,
                            weight_decay=self.cfg.weight_decay
                            )
            
        self.loss_function = F.mse_loss
        # self.ssim_score = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        # self.trainer.train_dataloader  # now accessible :)
        num_devices = (
            torch.cuda.device_count()
            if self.trainer.num_devices == -1
            else int(self.trainer.num_devices)
        )
        
        # self.trainer.reset_train_dataloaders(self)
        if self.cfg.max_epochs > 0:
            total_steps = (
                (1 + len(self.trainer.datamodule.train_dataloader())
                // self.cfg.accumulate_grad_batches
                // num_devices)
                * self.cfg.max_epochs 
            )
        else:
            total_steps = self.cfg.max_steps
            
        # warmup_steps =self.cfg.warmup_step
        # scheduler = {
        #     "scheduler": CosineAnnealingLR(optimizer, total_steps, 1.18e-7),
        #     "interval": "step",  # runs per batch rather than per epoch
        #     "frequency": 1,
        #     "name": "learning_rate",
        # }
        
        scheduler = {
            "scheduler": WarmupCosineSchedule(
                optimizer, warmup_steps=self.cfg.warmup_steps, t_total=total_steps),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]
        # return optimizer

    def get_loss(self, batch, mode='train'):
        imgs, label, days, treatments = batch["image"], batch["label"], batch["days"], batch["treatments"]
        n_sess = label.shape[1]
        
        
        # b, c, s, h, w = imgs.shape
        b, s, c, h, w = imgs.shape
        s1_days, s2_days, s3_days, t_days = days[:, 0], days[:,1], days[:, 2], days[:, 3]
        
        if mode == 'train' and np.random.random_sample() > 0.5: 
            # i_tg = np.random.randint(s, size=b)
            # is_repeat = s2_days == s1_days
            # i_tg[is_repeat.detach().cpu().numpy()] = -1
            i_tg = torch.randint(0, s, (b,), device=self.device)
            # i_tg[(s2_days == s1_days) * (s3_days == s2_days)] = -1
            i_tg[(s2_days != s1_days) * (s3_days == s2_days)] = -1 if np.random.random_sample() > 0.5 else 0
            i_tg[(s2_days == s1_days) * (s3_days != s2_days)] = -1 if np.random.random_sample() > 0.5 else -2
            # i_tg[s3_days == s2_days] = -1
        else:
            # i_tg = -np.ones(b, dtype=np.int8)
            i_tg = -torch.ones((b,), dtype=torch.int8, device=self.device)

        
        treat1, treat2, treat3, treat_t = treatments[:,0], treatments[:,1], treatments[:,2], treatments[:,3]
        # intvs = [s1_days.to(device), s2_days.to(device), t_days.to(device)]
        # print(f'treat_cond: {treat_cond[0]}')
        intvs = [s1_days.to(torch.float32), s2_days.to(torch.float32), 
                 s3_days.to(torch.float32), t_days.to(torch.float32)]
        treat_cond = [treat1.to(torch.float32), treat2.to(torch.float32),  
                      treat3.to(torch.float32), treat_t.to(torch.float32)]
        
        gt_img = torch.cat([imgs[[i], j, :, :, :] for i, j in zip(range(b), i_tg)], dim=0)
        gt_label = torch.cat([label[[i], j, :, :] for i, j in zip(range(b), i_tg)], dim=0)
        t = torch.randint(1, self.diffusion.T + 1, [gt_img.shape[0]]) # , device=self.device
        w_tg = self.alphabar[t-1]
        
        xt, epsilon = self.diffusion.sample(gt_img.to(torch.float32), t)
        # t = self.rng.draw(img.shape[0]) # draw t/timesteps from uniform distribution
        # noised_x, target, t, weights = self.diffusion.sample(label.to(torch.float32), t)
        # for i, j in zip(range(b), i_tg):
        #     seq_imgs[i][:, j, :, :] = xt[i, :, :, :]  # nosie target image
        
        maskout_batch = (s3_days == t_days) 
        for i, j in zip(range(b), i_tg):
            if maskout_batch[i]:
                imgs[i, :, :, :, :] = 0.
                label[i, :, :, :] = 0
            label[i, j, :, :] = gt_label[i, :, :]
            imgs[i, j, :, :, :] = xt[i, :, :, :]  # nosie target image
        # xt = torch.cat((seq_imgs), dim=0).transpose(1,2).reshape(b, s*c, h, w).contiguous() 
        xt = imgs.reshape(b, s*c, h, w).contiguous() 
        t = t.view(gt_img.shape[0]).to(self.device)
        
        # xt = torch.cat((cond_img, xt), dim=1)
        
        out = self.forward(xt.to(torch.float32), t.to(torch.float32), 
                           intv_t=intvs, treat_code=treat_cond, 
                           i_tg=i_tg)
        
        # Compute loss and backprop
       
        loss_weigths = torch.sum(label, dim=1, keepdim=True) # range 0 -4
        loss_weigths = loss_weigths * torch.exp(-loss_weigths)
        # loss_weigths = torch.clamp(F.conv2d(loss_weigths, self.dilation_filters.to(loss_weigths.device), padding='same'), 0, 1)
        # loss_weigths = torch.clamp(F.conv2d(loss_weigths, self.dilation_filters.to(loss_weigths.device), padding='same'), min=0.886)
        loss_weigths = F.conv2d(loss_weigths, self.dilation_filters.to(loss_weigths.device), padding='same') + 1.
        
        img_pred, mask_pred = out[:, 4:7, :, :], out[:, 0:4, :, :]
        
        loss1 = torch.mean(loss_weigths * (img_pred - epsilon)**2)
        mse = self.loss_function(img_pred, epsilon) # without weights on tumor
        
        dice_loss = self.dice(mask_pred, label).squeeze() # all segementaed masks b, 4, 1, 1
        
        w_tg = torch.from_numpy(w_tg).to(self.device) # (b, )
        # dice_loss = dice_loss * w_tg.view(b, 1)  # weighted the loss one more time, w_tg ** 3 for target image, but for refence image only appply w_tg
        # weighted future tumor loss based on nosized level
        for i, j in zip(range(b), i_tg):
            # dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i])
            if maskout_batch[i]:
                loss_ij = dice_loss[i, j] * torch.sqrt(w_tg[i])
                dice_loss[i, :] = 0.
                dice_loss[i, j] = loss_ij  # weight target image loss, w_tg ** 3
            else:
                dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i]) # w_tg[i]**2  # weight target image loss, w_tg ** 3
            
        # w_dims = (b,) + tuple((1 for _ in dice_loss.shape[1:])) 
        # dice_loss = dice_loss * w_tg.view(b, 1)  # weighted the loss one more time, w_tg ** 3 for target image, but for refence image only appply w_tg
        
        loss = loss1 + torch.mean(dice_loss) * self.cfg.aux_loss_w
        
        # mask_pred = F.sigmoid(mask_pred)
        mask_pred = torch.sigmoid(mask_pred)
        mask_pred = (mask_pred > 0.5) * 1  # fix threshold for segment mask 0.5
        self.dice_metric(mask_pred, label)
        dice_last =  self.dice_metric.aggregate() # only mean 4 mask dices
        self.dice_metric.reset()
        # if mode == 'train':
        #     self.dice_metric(mask_pred, label)
        #     dice_last =  self.dice_metric.aggregate() # only mean 4 mask dices
        #     self.dice_metric.reset()
        # else: 
        #     self.dice_metric(mask_pred[:, 3:4,:, :], label[:, 3:4, :, :])
        #     dice_last = self.dice_metric.aggregate()#.item() # only last masks 
        #     self.dice_metric.reset()
        
        return loss, mse, dice_last
        
    def training_step(self, batch, batch_idx):
        loss, mse, dice_seg = self.get_loss(batch, mode='train')
        self.log("train_loss", loss,  sync_dist=True, on_epoch=True, prog_bar=True) # on_epoch=False default
        self.log("train_mse", mse,  sync_dist=True, on_epoch=False, prog_bar=False) # on_epoch=False default
        self.log("train_dice", dice_seg,  sync_dist=True, on_epoch=False, prog_bar=False) # on_epoch=False default
        return {"loss": loss, "mse": mse, "dice_seg": dice_seg}

    def validation_step(self, batch, batch_idx):
        loss, mse, dice = self.get_loss(batch, mode='val')
        self.val_step_outputs.append({"val_loss": loss})
        self.log("val_loss", loss.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        self.log("val_mse", mse.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        self.log("val_dice", dice.item(), sync_dist=True, prog_bar=False) # on_epoch=True default
        return {"val_loss": loss, "val_mse": mse, "val_dice": dice}


class MyCallback(Callback):
    def __init__(self, batch, config):
        super().__init__()
        self.batch = batch
        self.cfg = config
        # self.img = self.batch["image"]
        # b, s, c, h, w
        img_label = torch.cat([self.batch["image"], self.batch["label"].unsqueeze(2)], dim=2)
        days = self.batch["days"]
        treatments = self.batch["treatments"]
        # n_sess = img_label.shape[1]
        # self.val_labels = img_label[:, :, n_sess-1, :, :]
        self.val_labels = img_label[:, :, -1, :, :] # 4sess, h, w
        self.img_cond = img_label[:, :-1, :-1, :, :] # c-modal, 3sess, h, w
        self.img_for_noise = img_label[:, -1, :-1, :, :]  # c-modal, 1sess,  h, w
        b, s, c, h, w = self.img_cond.shape
        self.img_cond = self.img_cond.reshape(b, s*c, h, w).contiguous()
        self.gt_preimg = img_label#[:, :, :-1, :, :]
        
        s1_days, s2_days, s3_days, t_days = days[:, 0], days[:,1], days[:, 2], days[:, 3]
        # intvs = [s1_days.to(device), s2_days.to(device), t_days.to(device)]
        # print(f'treat_cond: {treat_cond[0]}')
        self.intvs = [s1_days.to(torch.float32), s2_days.to(torch.float32), 
                      s3_days.to(torch.float32), t_days.to(torch.float32)]
        
        
        treat1, treat2, treat3, treat_t = treatments[:,0], treatments[:,1], treatments[:,2], treatments[:,3]
        self.treat_cond = [treat1.to(torch.float32), treat2.to(torch.float32),  
                      treat3.to(torch.float32), treat_t.to(torch.float32)]
        
        # zero_mask = torch.zeros_like(self.val_labels).unsqueeze(2)
        noise = torch.randn((self.img_for_noise.shape))
        self.val_imgs = torch.cat([self.img_cond, noise], dim=1)

        # self.diffusion = LinearDiffusion(T=1000)
        # self.diffusion = GaussianDiffusion(T=1000, schedule='linear')
        self.diffusion = GaussianDiffusion(T=int(self.cfg.max_T), schedule=self.cfg.ddpm_schedule)
        # self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
    
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # clean up artifacts cache
        # c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
        # c.cleanup(wandb.util.from_human_size("0GB"))
        
        mean_val_loss = torch.stack([torch.tensor(x["val_loss"].clone().detach()) for x in pl_module.val_step_outputs]).mean()
        # pl_module.log("val_avg_dice", mean_val_dice, sync_dist=True )
        # pl_module.log("val_avg_loss", mean_val_loss, sync_dist=True)
        if pl_module.best_val_loss is None:
            pl_module.best_val_loss = mean_val_loss
            pl_module.best_val_epoch = pl_module.current_epoch
        elif mean_val_loss < pl_module.best_val_loss:
            pl_module.best_val_loss = mean_val_loss
            pl_module.best_val_epoch = pl_module.current_epoch
            
        if pl_module.global_rank == 0:
            print("on_validation_epoch_end...")
            print(
                f"current epoch: {pl_module.current_epoch} "
                f"current mean loss: {mean_val_loss:.4f}"
                f"\nbest mean loss: {pl_module.best_val_loss:.4f} "
                f"at epoch: {pl_module.best_val_epoch}"
            )
            # self.log("best mean loss:",pl_module.best_val_loss)
            # self.log("at best epoch:", pl_module.best_val_epoch)
        val_imgs = self.val_imgs.to(device=pl_module.device) # img[:, 0:9, :, :].unsqueeze(1)
        # val_labels = self.val_labels.to(device=pl_module.device) # img[:, 9:12, :, :]
        # timesteps = [t.to(device=pl_module.device) for t in self.timesteps]
        # Get model prediction
        intvs = [intv.to(device=pl_module.device) for intv in self.intvs]
        treat_cond = [treat.to(device=pl_module.device) for treat in self.treat_cond]
        preds, aux_out = self.diffusion.TaDiff_inverse2(pl_module, 
                                        start_t=self.cfg.max_T//1.5, #600, 
                                        steps=self.cfg.max_T//1.5, #600,
                                        x=val_imgs, 
                                        intv=intvs, 
                                        treat_cond=treat_cond,
                                        # days=self.days.to(device=pl_module.device), 
                                        # treat=self.treatments.to(device=pl_module.device),
                                        device=pl_module.device)
        # Log the images as wandb Image
        aux_out = torch.sigmoid(aux_out)
        
        columns = ['days/tr-1', 'days/r-2', 'days/tr-3', 'tg-days/tr']
        my_data = [[f'{d1}-{tr1}', f'{d2}-{tr2}', f'{d3}-{tr3}', f'{td}-{ttr}'] for d1, tr1, d2, tr2, d3, tr3, td, ttr in 
                   list(zip(intvs[0], treat_cond[0], 
                            intvs[1], treat_cond[1], 
                            intvs[2], treat_cond[2], 
                            intvs[3], treat_cond[3]))]
        # data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
         
        trainer.logger.log_table(key='test_samples', columns = columns, data = my_data)
        
        trainer.logger.log_image(key="label", 
                                    images=[self.gt_preimg[0, -2, 3, :, :].cpu().detach().numpy(), 
                                            self.gt_preimg[0, -1, 3, :, :].cpu().detach().numpy(),
                                            aux_out[0, 3, :, :].cpu().detach().numpy(),
                                            aux_out[0, 2, :, :].cpu().detach().numpy(),
                                            ],
                                    caption=[f'input:day{intvs[2][0]}-tr{treat_cond[2][0]}', f'target:day{intvs[3][0]}-tr{treat_cond[3][0]}', "pred-mask-tg", "pred-mask-s3"]) # f'Ground Truth: {y_i}
        
        trainer.logger.log_image(key="Flair", 
                                    images=[self.gt_preimg[3, -2, 2, :, :].cpu().detach().numpy(), 
                                            self.gt_preimg[3, -1, 2, :, :].cpu().detach().numpy(), 
                                            preds[3, 2, :, :].cpu().detach().numpy(),
                                            aux_out[3, 3, :, :].cpu().detach().numpy()
                                            ],
                                    caption=[f'input:day{intvs[2][3]}-tr{treat_cond[2][3]}', f'target:day{intvs[3][3]}-tr{treat_cond[3][3]}', "pred-img", "pred-mask-tg"])
        
        trainer.logger.log_image(key="T1c", 
                                    images=[self.gt_preimg[1, -2, 1,  :, :].cpu().detach().numpy(), 
                                            self.gt_preimg[1, -1, 1,  :, :].cpu().detach().numpy(),
                                            preds[1, 1, :, :].cpu().detach().numpy(),
                                            aux_out[1, 3, :, :].cpu().detach().numpy(),
                                            ],
                                    caption=[f'input:day{intvs[2][1]}-tr{treat_cond[2][1]}', f'target:day{intvs[3][1]}-tr{treat_cond[3][1]}', "pred-img", "pred-mask-tg"])
        
        trainer.logger.log_image(key="T1", 
                                    images=[self.gt_preimg[2, -2, 0,  :, :].cpu().detach().numpy(), 
                                            self.gt_preimg[2, -1, 0,  :, :].cpu().detach().numpy(),
                                            preds[2, 0, :, :].cpu().detach().numpy(),
                                            aux_out[2, 3, :, :].cpu().detach().numpy(),
                                            ],
                                    caption=[f'input:day{intvs[2][2]}-tr{treat_cond[2][2]}', f'target:day{intvs[3][2]}-tr{treat_cond[3][2]}', "pred-img", "pred-mask-tg"])

        pl_module.val_step_outputs.clear()
         
# # trainer = Trainer(callbacks=[MyPrintingCallback()])