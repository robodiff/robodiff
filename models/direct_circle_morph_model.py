import torch
import numpy as np
from .base_models import BaseMorphModel
from .sd_utils import * 
import IPython
#%%
class DirectCircleMorphModel(BaseMorphModel):
    def __init__(self, scene=None,
                        morphMask=None,
                        holeSize=0.2,
                        target_hole_area=None,
                        max_hole_size=None,
                        genome=None,
                        optimizer=None,
                        gui=None,
                        device=None,
                        fringe_softness=2,
                        secondary_loss=None,
                        secondary_losses=[],
                        erode_loss_amount=1e-4,
                        erode_target_amount=0.5,
                        rotational_moment_amount=1.0,
                        mask_alignment_amount=1.0,
                        circle_mask_size=0.25,
                        reset_patches=False,
                        **kwargs):
        super().__init__(genome=genome, optimizer=optimizer, gui=gui, device=device, **kwargs)

        # old behavior: don't optimize hole size:
        # setup after initializing the genome / optimizer of the super class
        if "hole_size" not in self._genome:
            self._genome["hole_size"] = torch.tensor([holeSize]*self._genome["hole_x"].shape[0], device=device)

        self._scene = scene
        self._morphMask = morphMask
        self._hole_size = holeSize
        self._target_hole_area = target_hole_area
        self._max_hole_size = max_hole_size
        self._fringe_softness = fringe_softness
        self._reset_patches = reset_patches
        self._secondary_losses = secondary_losses + [secondary_loss]
        self._erode_loss_amount = erode_loss_amount
        self._erode_target_amount = erode_target_amount
        self._rotational_moment_amount = rotational_moment_amount
        self._mask_alignment_amount = mask_alignment_amount
        self._circle_mask_size = circle_mask_size
        self._step_idx = 0
        self._active_holes_log = None
        self._active_mass = None

        self._final_tensor = None

        if self._morphMask == "platypus":
            # platypus
            # url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Platypus_illustration.jpg"

            # guitar pick
            url = "https://cdn.shopify.com/s/files/1/0571/6549/products/cell_standard_black_48a21706-e3fd-496a-a5e0-0a62fedf970c.jpg"

            from PIL import Image
            import requests
            from io import BytesIO
            # import matplotlib.pyplot as plt

            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).resize((64,64))
            img_np = np.array(img)[:,:, 0] < 255
            img_np[:, :-6] = img_np.T[:, ::-1][:, 6:]
            img_np[:, -6:] = False
            # IPython.embed() 

            self._cached_img_mask = torch.tensor(img_np, device=device)

    def get_summary_string(self):
        return super().get_summary_string(name="Morph Direct Circle", truncate=10)

    def normalize_hole_sizes(self, orig_hole_sizes):
        if self._target_hole_area is not None:
            tha_c = self._target_hole_area / np.pi

            hole_sizes_sq = torch.pow(orig_hole_sizes, 2)
            curr_hole_area_prime = hole_sizes_sq.sum()

            if ( self._genome["hole_x"].shape[0] != orig_hole_sizes.shape[0] ):
                curr_hole_area = curr_hole_area_prime * self._genome["hole_x"].shape[0]
            else:
                curr_hole_area = curr_hole_area_prime

            area_overspill = curr_hole_area / tha_c

            if area_overspill > 1.0:
                return torch.sqrt(hole_sizes_sq/area_overspill)
            else:
                return orig_hole_sizes
        else:
            return orig_hole_sizes

    def enforce_max_hole_size(self, arr):
        if self._max_hole_size is not None:
            return torch.min(arr, torch.ones_like(arr)*self._max_hole_size)
        else:
            return arr
            
    def generate(self):
        _c_x_t = self._genome["hole_x"]
        c_x_t = _c_x_t
        c_x_t = torch.where(torch.isfinite(_c_x_t), _c_x_t, torch.ones_like(_c_x_t)*100)

        _c_y_t = self._genome["hole_y"]
        c_y_t = _c_y_t
        c_y_t = torch.where(torch.isfinite(_c_y_t), _c_y_t, torch.ones_like(_c_y_t)*100)

        __hole_size_t = torch.max(self._genome["hole_size"], torch.zeros_like(self._genome["hole_size"]))
        _hole_size_t = self.enforce_max_hole_size(__hole_size_t)
        hole_size_t = self.normalize_hole_sizes(torch.where(torch.isfinite(_hole_size_t), _hole_size_t, torch.zeros_like(_hole_size_t)))

        x_t = torch.linspace(0, 1, self._scene.w_count, requires_grad=True, device=self.device)
        xx_t, yy_t = torch.meshgrid(x_t, x_t)
        xx_t = xx_t[:, :, None].repeat((1, 1, c_x_t.shape[0]))
        yy_t = yy_t[:, :, None].repeat((1, 1, c_x_t.shape[0]))
        if self._morphMask == "triangleA":
            # base side down triangle
            H = .50
            B = 1.0
            morph_mask_t = (yy_t[:,:,0] < 2*H/B * xx_t[:,:,0])  & (yy_t[:,:,0] < 2 * H *  (1-(xx_t[:,:,0]/B)))
            # IPython.embed()
        elif self._morphMask == "triangleB":
            # point side down triangle
            H = .50
            B = 1.0
            morph_mask_t = (yy_t[:,:,0] < H) & (H - yy_t[:,:,0] < 2*H/B * xx_t[:,:,0])  & (H - yy_t[:,:,0] < 2 * H *  (1-(xx_t[:,:,0]/B)))
        elif self._morphMask == "triangleC":
            morph_mask_t = get_star(xx_t, yy_t, 3, 2, 0.4)
        elif self._morphMask == "star5":
            morph_mask_t = get_star(xx_t, yy_t, 5, 3, 0.5)
        elif self._morphMask == "star5small":
            morph_mask_t = get_star(xx_t, yy_t, 5, 3, 0.4)
        elif self._morphMask == "star6":
            morph_mask_t = get_star(xx_t, yy_t, 6, 3, 0.5)
        elif self._morphMask == "star7":
            morph_mask_t = get_star(xx_t, yy_t, 7, 3, 0.5)
        elif self._morphMask == "star8":
            morph_mask_t = get_star(xx_t, yy_t, 8, 3, 0.5)
        elif self._morphMask == "star9a":
            morph_mask_t = get_star(xx_t, yy_t, 9, 5, 0.5)
        elif self._morphMask == "star9b":
            morph_mask_t = get_star(xx_t, yy_t, 9, 3, 0.5)

            
        elif self._morphMask == "circle":
            # circle mask
            radius = 0.35
            offset_x = 0.5
            offset_y = 0.35
            morph_mask_t = torch.pow(yy_t[:,:,0] - offset_y, 2) +  torch.pow(xx_t[:,:,0] - offset_x, 2)  < radius**2
        elif self._morphMask == "platypus":
            morph_mask_t = self._cached_img_mask
        elif self._morphMask == "none" or self._morphMask is None:
            # full rectangle
            morph_mask_t = yy_t[:,:,0] == yy_t[:,:,0]
        else:
            raise ValueError(f"morphMask is not recognized {self._morphMask}")

        dx_t = torch.pow(torch.abs(xx_t - c_x_t)+1e-9, 2)
        dy_t = torch.pow(torch.abs(yy_t - c_y_t)+1e-9, 2)

        # dx_t = torch.pow(torch.abs(xx_t - c_x_t), 2)
        # dy_t = torch.pow(torch.abs(yy_t - c_y_t), 2)

        dist_t = torch.sqrt(dx_t + dy_t) / (hole_size_t+1e-9)
        frac_towards_center_t, actives_holes = dist_t.min(axis=2)
        with torch.no_grad():
            self._active_holes_log = actives_holes.detach().unique()
        # IPython.embed()

        # Apply the soft fringe, if using.
        # when softness == 0,
        # particles with a mass below 0.5 are set to 0.
        # and particles with a mass above, are clipped to a mass of 1
        # this is effectively a linear interpolation model.
        if self._fringe_softness == 0:
            pre_brightness_t = torch.where(
                frac_towards_center_t < .50, # inner 50% -> 0
                frac_towards_center_t * 0.0,
                frac_towards_center_t)
            brightness_t = torch.where( # threshold to 1.0
                pre_brightness_t < 1.0,
                pre_brightness_t,
                torch.ones_like(pre_brightness_t))
        else:
            # otherwise, particles that are inside of the patches are raise to the power of the fringe softness.
            # particles outside of the patches are clipped to 1.
            # This method works for arbitrary soft fringe values.
             brightness_t = torch.where(
                frac_towards_center_t < 1.0, # near one hole,
                torch.pow(torch.abs(frac_towards_center_t), self._fringe_softness), # apply transformation
                torch.ones_like(frac_towards_center_t)) # else full mass

        # IPython.embed()
        
        morph_masked_brightness_t = brightness_t * morph_mask_t

        output_t = torch.nn.functional.interpolate(morph_masked_brightness_t[None, None, :, :], (self._scene.w_count, self._scene.w_count))

        output_t = output_t[0,0][:, :self._scene.h_count].flatten()

        self._final_tensor = output_t
        with torch.no_grad():
            # ROBUSTNESS_STUDY_CONSIDER_EXTENDING
            self._active_mass = (torch.where(output_t.detach() > 0.1, output_t.detach(), torch.zeros_like(output_t.detach())).sum()/(self._scene.w_count * self._scene.h_count)).item()
        # IPython.embed()
        return {
            'mass':output_t.detach().cpu().numpy()
        }

    def step(self):
        super().step()
        if self._optimizer is not None and self._reset_patches:
            with torch.no_grad():
                hs = self._genome["hole_size"]
                hx = self._genome["hole_x"]
                hy = self._genome["hole_y"] 
                # pos = torch.stack([hx - 0.5, hy - 0.35])
                # box = torch.tensor([0.5, 0.35])[:, None].repeat([1, hx.shape[0]])
                # dist_to_robot_rect = sdBox(pos, box )
                # inactive_holes_size = hs <= 0
                # inactive_holes_location = dist_to_robot_rect > hs/2

                # inactive_holes = inactive_holes_location | inactive_holes_size

                num_holes = hx.shape[0]
                uniques, counts = torch.cat((torch.arange(num_holes), self._active_holes_log)).unique(return_counts=True)
                inactive_holes = uniques[counts==1]
                num_inactive = inactive_holes.shape[0]

                # reset the holes
                self._genome["hole_size"][inactive_holes] = 0.06 # reset to base size 
                self._genome['hole_x'][inactive_holes]  = torch.rand(num_inactive)
                self._genome['hole_y'][inactive_holes]  = torch.rand(num_inactive)

                # reset the adam momentum.
                for paramGroupState in self._optimizer.state.values():
                    paramGroupState['exp_avg'][inactive_holes] = 0
                    paramGroupState['exp_avg_sq'][inactive_holes] = 0
                print(f"Active mass: {self._active_mass} | resetting {num_inactive} holes")
                # print(f"resetting {num_inactive} holes ({inactive_holes_size.sum()}s, {inactive_holes_location.sum()}loc)\n\n")

    def backward(self, **kwargs):
        super().backward(**kwargs)
        
        mass_grad = kwargs['mass']
        for secondary_loss in self._secondary_losses:
            if secondary_loss is not None:
                print(f"applying {secondary_loss} loss")
                if secondary_loss == "erode":
                    mass_grad +=  self._erode_loss_amount
                elif secondary_loss == "erodeProportional":
                    f = lambda x: np.tanh((x-self._erode_target_amount) * 10)
                    pressure_fraction = f(self._active_mass)
                    mass_grad_range = mass_grad.max() - mass_grad.min()
                    mass_grad += mass_grad_range * self._erode_loss_amount * pressure_fraction
                    print(f"Applying erosion pressure fraction {pressure_fraction} during backpropagation")
                elif secondary_loss == "rotationalMoment":
                    design_t = torch.tensor(self._final_tensor.detach().cpu().numpy().reshape(1,1,self._scene.w_count, self._scene.h_count), requires_grad=True)
                    x = torch.linspace(0, 1, self._scene.w_count)
                    yy,xx = torch.meshgrid([x,x])
                    xx = xx[:self._scene.h_count].T.reshape(1, 1, self._scene.w_count, self._scene.h_count)
                    yy = yy[:self._scene.h_count].T.reshape(1, 1, self._scene.w_count, self._scene.h_count)
                    xmean = (design_t * xx).sum() / design_t.sum()
                    ymean = (design_t * yy).sum() / design_t.sum()
                    xmean = xmean.item()
                    ymean = ymean.item() # / 0.7
                    dx = torch.pow(xx - xmean, 2) + torch.pow( yy - ymean, 2)
                    dx_thresholded = torch.where(dx > 0.25, dx * 2 - 0.25, dx) # apply harsher penalty for particles farther away.
                    rotational_moment = (dx_thresholded * design_t).sum()
                    rotational_moment.backward()
                    print(f"rotational moment is: {rotational_moment.item()}")
                    mass_grad += design_t.grad.flatten().detach().cpu().numpy() * self._rotational_moment_amount
                elif secondary_loss == "circleMask":
                    design_t = torch.tensor(self._final_tensor.detach().cpu().numpy().reshape(1,1,self._scene.w_count, self._scene.h_count), requires_grad=True)
                    reward_mask = torch.tensor(np.fromfunction(lambda x,y: get_circle(x,y, circle_size=self._circle_mask_size), (self._scene.w_count, self._scene.h_count))).reshape(1,1,self._scene.w_count, self._scene.h_count)
                    mask_alignment = (reward_mask * design_t).sum()
                    mask_alignment.backward()
                    mass_grad += design_t.grad.flatten().detach().cpu().numpy() * get_amount(self._mask_alignment_amount, self._step_idx)
        self._final_tensor.backward(self.validate_grads('mass', mass_grad))
        self._step_idx += 1

def get_amount(x, idx):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return x[idx] if idx < len(x) else x[len(x)-1]
    return x


def get_circle(x,y, circle_size=0.25):
    dx = x/x.shape[0] - circle_size
    dy = y/x.shape[0] - circle_size
    dist = np.power(dx, 2) + np.power(dy, 2)
    # return (dist >  circle_size**2)  # loss is amount of material outside of circle. 
    return (dist <= circle_size**2) * -2 + 1   # reward material inside of the circle, and penalize material outside.

def get_star(x, y, n, m, r):
    y_translate = np.sin( np.ceil(n/2) * np.pi * 2 / n + np.pi/2) * r

    p = torch.stack([x[:,:,0].flatten(), y[:,:,0].flatten()]) - torch.tensor([[0.5],[y_translate * -1]]).float()

    signed_dist_t = sdStarTorch(p, r, n, m)
    return (signed_dist_t < 0).reshape(x.shape[:2]) # which particles are inside the shape.
