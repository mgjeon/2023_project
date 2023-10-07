import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from nnf2.train.model import BModel, jacobian
from pytorch_lightning import LightningModule

import wandb


class NF2Module(LightningModule):
    def __init__(
        self,
        validation_settings,
        dim=256,
        lambda_b={"start": 1e3, "end": 1, "iterations": 1e5},
        lambda_div=0.1,
        lambda_ff=0.1,
        lr_params={"start": 5e-4, "end": 5e-5, "decay_iterations": 1e5},
        meta_path=None,
        initialization=False,
        **kwargs,
    ):
        super().__init__()

        self.validation_settings = validation_settings

        model = BModel(3, 3, dim)
        self.model = model

        self.lr_params = lr_params

        if meta_path:
            state_dict = (
                torch.load(meta_path)["model"].state_dict()
                if meta_path.endswith("nf2")
                else torch.load(meta_path)["m"]
            )
            model.load_state_dict(state_dict)
            logging.info(f"Loaded meta state: {meta_path}")

        if isinstance(lambda_b, dict):
            self.register_buffer(
                "lambda_B", torch.tensor(lambda_b["start"], dtype=torch.float32)
            )
            self.register_buffer(
                "lambda_B_gamma",
                torch.tensor(
                    (lambda_b["end"] / lambda_b["start"])
                    ** (1 / lambda_b["iterations"])
                    if lambda_b["iterations"] > 0
                    else 0,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "lambda_B_end", torch.tensor(lambda_b["end"], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "lambda_B", torch.tensor(lambda_b, dtype=torch.float32)
            )
            self.register_buffer("lambda_B_gamma", torch.tensor(1, dtype=torch.float32))
            self.register_buffer(
                "lambda_B_end", torch.tensor(lambda_b, dtype=torch.float32)
            )

        if initialization:
            self.register_buffer(
                "initialization", torch.tensor(initialization, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initialization", torch.tensor(initialization, dtype=torch.float32)
            )
            self.register_buffer(
                "lambda_div", torch.tensor(lambda_div, dtype=torch.float32)
            )
            self.register_buffer(
                "lambda_ff", torch.tensor(lambda_ff, dtype=torch.float32)
            )


    def configure_optimizers(self):
        parameters = list(self.model.parameters())

        if isinstance(self.lr_params, dict):
            lr_start = self.lr_params["start"]
            lr_end = self.lr_params["end"]
            decay_iterations = self.lr_params["decay_iterations"]
        else:
            lr_start = self.lr_params
            lr_end = self.lr_params
            decay_iterations = 1

        self.optimizer = torch.optim.Adam(parameters, lr=lr_start)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=(lr_end / lr_start) ** (1 / decay_iterations)
        )

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        boundary_batch = batch["boundary"]
        boundary_coords = boundary_batch["coords"]
        boundary_values = boundary_batch["values"]

        if self.initialization:
            boundary_b = self.model(boundary_coords)
            b_diff = torch.abs(boundary_b - boundary_values)
            b_diff = torch.mean(torch.nanmean(b_diff.pow(2), -1))

            loss = b_diff

            loss_dict = {"b_diff": b_diff}

            loss_dict["loss"] = loss
        else:
            boundary_coords.requires_grad = True
            random_coords = batch['random']
            random_coords.requires_grad = True
            n_boundary_coords = boundary_coords.shape[0]
            coords = torch.cat([boundary_coords, random_coords])

            b = self.model(coords)

            boundary_b = b[:n_boundary_coords]
            b_diff = torch.abs(boundary_b - boundary_values)
            b_diff = torch.mean(torch.nanmean(b_diff.pow(2), -1))

            divergence_loss, force_free_loss = calculate_loss(b, coords)
            divergence_loss, force_free_loss = divergence_loss.mean(), force_free_loss.mean()

            loss = b_diff * self.lambda_B + \
                   divergence_loss * self.lambda_div + \
                   force_free_loss * self.lambda_ff
            
            loss_dict = {"b_diff": b_diff, "divergence_loss": divergence_loss, "force_free_loss": force_free_loss}

            loss_dict["loss"] = loss 

        return loss_dict

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.lambda_B > self.lambda_B_end:
            self.lambda_B *= self.lambda_B_gamma

        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()

        self.log("Learning Rate", self.scheduler.get_last_lr()[0])
        self.log("Lambda B", self.lambda_B)

        self.log("train/loss", outputs["loss"])
        self.log("Training Loss", {k: v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx):
        if dataloader_idx == 0:
            coords = batch
            coords.requires_grad = True

            b = self.model(coords)

            jac_matrix = jacobian(b, coords)
            dBx_dx = jac_matrix[:, 0, 0]
            dBy_dx = jac_matrix[:, 1, 0]
            dBz_dx = jac_matrix[:, 2, 0]
            dBx_dy = jac_matrix[:, 0, 1]
            dBy_dy = jac_matrix[:, 1, 1]
            dBz_dy = jac_matrix[:, 2, 1]
            dBx_dz = jac_matrix[:, 0, 2]
            dBy_dz = jac_matrix[:, 1, 2]
            dBz_dz = jac_matrix[:, 2, 2]
            #
            rot_x = dBz_dy - dBy_dz
            rot_y = dBx_dz - dBz_dx
            rot_z = dBy_dx - dBx_dy
            #
            j = torch.stack([rot_x, rot_y, rot_z], -1)
            div = torch.abs(dBx_dx + dBy_dy + dBz_dz)

            return {"b": b.detach(), "j": j.detach(), "div": div.detach()}

        elif dataloader_idx == 1:
            boundary_coords = batch["coords"]
            boundary_values = batch["values"]

            b = self.model(boundary_coords)
            b_diff = torch.abs(b - boundary_values)
            b_diff = torch.mean(torch.nanmean(b_diff.pow(2), -1))

            return {"b_diff": b_diff.detach()}

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0:
            return

        # data loader 0
        outputs = outputs_list[0]

        b = torch.cat([o["b"] for o in outputs])
        j = torch.cat([o["j"] for o in outputs])
        div = torch.cat([o["div"] for o in outputs])

        b_norm = b.pow(2).sum(-1).pow(0.5)
        j_norm = j.pow(2).sum(-1).pow(0.5)

        jxb = torch.cross(j, b, dim=-1)
        jxb_norm = jxb.pow(2).sum(-1).pow(0.5)

        eps = 1e-7

        angle = jxb_norm / (b_norm * j_norm)
        sig = torch.asin(torch.clip(angle, -1.0 + eps, 1.0 - eps))
        sig = torch.abs(torch.rad2deg(sig))
        weighted_sig = np.average(sig.cpu().numpy(), weights=j_norm.cpu().numpy())
        sigma_J = np.average(angle.cpu().numpy(), weights=j_norm.cpu().numpy())
        theta_J = np.rad2deg(np.arcsin(sigma_J))

        div_loss = div / (b_norm + eps)
        div_loss = div_loss.mean()

        ff_loss = jxb_norm / (b_norm + eps)
        ff_loss = ff_loss.mean()

        # data loader 1
        outputs = outputs_list[1]

        b_diff = torch.stack([o["b_diff"] for o in outputs]).mean()

        # plot
        b_cube = b.reshape(*self.validation_settings["cube_shape"], 3).cpu().numpy()
        self.plot_sample(b_cube)

        # log
        self.log("Validation B_diff", b_diff)
        self.log("Validation DIV", div_loss)
        self.log("Validation FF", ff_loss)
        self.log("Validation Sigma", weighted_sig)
        self.log("Validation theta_J", theta_J)

        return {
            "progress_bar": {
                "b_diff": b_diff,
                "div": div_loss,
                "ff": ff_loss,
                "sigma": weighted_sig,
                "theta_J": theta_J,
            },
            "log": {
                "val/b_diff": b_diff,
                "val/div": div_loss,
                "val/ff": ff_loss,
                "val/sigma": weighted_sig,
                "val/theta_J": theta_J,
            },
        }

    def plot_sample(self, b, n_samples=10):
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        heights = np.linspace(0, 1, n_samples) ** 2 * (b.shape[2] - 1)
        heights = heights.astype(np.int32)

        for i in range(3):
            for j, h in enumerate(heights):
                v_min_max = np.max(np.abs(b[:, :, h, i]))
                axs[i, j].imshow(
                    b[:, :, h, i].T,
                    cmap="gray",
                    vmin=-v_min_max,
                    vmax=v_min_max,
                    origin="lower",
                )
                axs[i, j].set_axis_off()

        for j, h in enumerate(heights):
            axs[0, j].set_title(f"{h:.01f}")

        fig.tight_layout()

        wandb.log({"Slices": fig})
        plt.close("all")

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.lambda_B *= self.lambda_B_gamma ** checkpoint["global_step"]
        self.lambda_B = max([self.lambda_B, self.lambda_B_end])

def calculate_loss(b, coords):
    jac_matrix = jacobian(b, coords)
    dBx_dx = jac_matrix[:, 0, 0]
    dBy_dx = jac_matrix[:, 1, 0]
    dBz_dx = jac_matrix[:, 2, 0]
    dBx_dy = jac_matrix[:, 0, 1]
    dBy_dy = jac_matrix[:, 1, 1]
    dBz_dy = jac_matrix[:, 2, 1]
    dBx_dz = jac_matrix[:, 0, 2]
    dBy_dz = jac_matrix[:, 1, 2]
    dBz_dz = jac_matrix[:, 2, 2]
    #
    divergence_loss = (dBx_dx + dBy_dy + dBz_dz).pow(2)
    #
    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    #
    j = torch.stack([rot_x, rot_y, rot_z], -1)
    jxb = torch.cross(j, b, -1)
    force_free_loss = jxb.pow(2).sum(-1) / (b.pow(2).sum(-1) + 1e-7)
    return divergence_loss, force_free_loss

def save(save_path, model, data_module, config):
    save_state = {
        "model": model,
        "cube_shape": data_module.cube_shape,
        "b_norm": data_module.b_norm,
        "spatial_norm": data_module.spatial_norm,
        "meta_data": data_module.meta_data,
        "config": config,
        "Mm_per_pixel": data_module.Mm_per_pixel,
    }
    torch.save(save_state, save_path)
