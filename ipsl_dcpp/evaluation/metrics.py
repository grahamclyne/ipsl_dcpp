import torch
# this computes the metrics for evaluating forecasts
# torchmetrics convention ?

# metrics we want :
# - ensemble mean rmse
# - ensemble var
# - unbiased spread skill ratio estimation
# classifier score
# CRPS
# todo:
# forecast activity with climatology
# classifier score

from torchmetrics import Metric

class EnsembleMetrics(Metric):
    '''
    metrics for ensemble prediction including:
    - ensemble mean rmse
    - ensemble spread
    - spread skill ratio (bias-corrected formula)
    - spread skill ratio (bias-free formula with residual)
    - CRPS
    '''
    def __init__(self, dataset, level_shape=(5, 13), surface_shape=(4, 9)):
        # remember to call super
        super().__init__()
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("nmembers", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("err_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
     #   self.add_state("err_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")
        self.add_state("var_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
     #   self.add_state("var_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")
        self.add_state("norm_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
     #   self.add_state("norm_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")
        self.add_state("prederr_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
     #   self.add_state("prederr_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")

        # CRPS
        self.add_state("mae_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
   #     self.add_state("mae_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")

        self.add_state("disp_surface", default=torch.zeros(surface_shape), dist_reduce_fx="sum")
   #     self.add_state("disp_level", default=torch.zeros(level_shape), dist_reduce_fx="sum")


        self.add_state("lat_coeffs", default=dataset.lat_coeffs_equi)

        #self.lat_coeffs = 

        self.display_surface_metrics = [('T2m', 2, 0),
                                ('U10', 0, 0),
                                ('SP', 3, 0)]

        # self.display_level_metrics = [('Z500', 0, 7),
        #                       ('Q700', 4, 9),
        #                       ('T850', 3, 10)]

    def wmse(self, x, y=0): # weighted mse error
        return (x - y).pow(2).mul(self.lat_coeffs).nanmean((-2, -1))

    def wmae(self, x, y=0):
        return (x - y).abs().mul(self.lat_coeffs).nanmean((-2, -1))

    def wvar(self, x, dim=1): # weighted variance along axis
        return x.var(dim).mul(self.lat_coeffs).nanmean((-2, -1))
    
    def update(self, batch, preds) -> None:
        # inputs to this function should be denormalized
        if isinstance(preds, list):
            preds = {k:torch.stack([x[k] for x in preds], dim=1) for k in preds[0].keys()}

        self.nsamples += batch['next_state_surface'].shape[0]
        self.nmembers += preds['next_state_surface'].shape[0] * preds['next_state_surface'].shape[1] # total member predictions
        print(self.nmembers)
        print(self.nsamples)
        print(batch['next_state_surface'].shape)
        print(preds['next_state_surface'].shape)
        avg_preds = {k:v.mean(1) for k, v in preds.items()}
        print(avg_preds['next_state_surface'].shape)
      #  self.err_level += self.wmse(batch['next_state_level'] - avg_preds['next_state_level']).sum(0) # 2 dimensions remaining
        self.err_surface += self.wmse(batch['next_state_surface'] - avg_preds['next_state_surface']).nansum(0)
        self.var_surface += self.wvar(preds['next_state_surface']).sum(0)
      #  self.var_level += self.wvar(preds['next_state_level']).sum(0)

        # log norm for unbiased sskill ratio estimation
     #   self.norm_level += self.wmse(preds['next_state_level'] - batch['pred_state_level'].unsqueeze(1)).mean(1).sum(0) # average over nmembers
        self.norm_surface += self.wmse(preds['next_state_surface'] - batch['next_state_surface'].unsqueeze(1)).nanmean(1).sum(0) # average over nmembers

        # in the unbiased estimation, the err is given by next - pred
  #      self.prederr_level += self.wmse(batch['next_state_level'] - batch['pred_state_level']).sum(0)
        #self.prederr_surface += self.wmse(batch['next_state_surface'] - batch['pred_state_surface']).sum(0)

        # for CRPS
        self.mae_surface += self.wmae(preds['next_state_surface'] - batch['next_state_surface'].unsqueeze(1)).nanmean(1).sum(0)
     #   self.mae_level += self.wmae(preds['next_state_level'] - batch['next_state_level'].unsqueeze(1)).mean(1).sum(0)

        # for dispersion, we unsqueeze on different values, then broadcast and average
        self.disp_surface += self.wmae(preds['next_state_surface'].unsqueeze(1) - preds['next_state_surface'].unsqueeze(2)).nanmean((1, 2)).sum(0)
    #    self.disp_level += self.wmae(preds['next_state_level'].unsqueeze(1) - preds['next_state_level'].unsqueeze(2)).mean((1, 2)).sum(0)


    def compute(self) -> torch.Tensor:
        # compute final result
        nmembers = self.nmembers / self.nsamples
        spskr_coeff = (1 + 1/nmembers)**.5
        # maybe we should define the set of variables we are interested in ?
        surface_metrics = dict(
            err = self.err_surface / self.nsamples,
            var = self.var_surface / self.nsamples,

            norm = self.norm_surface/self.nsamples,
            
            spskr = spskr_coeff * (self.var_surface / self.err_surface).sqrt(),

            unbiased_spskr = (self.norm_surface / self.prederr_surface).sqrt(),
            
            crps = (self.mae_surface - .5*self.disp_surface)/self.nsamples,
            )
        
        # level_metrics = dict(
        #     err = self.err_level / self.nsamples,
        #     var = self.var_level / self.nsamples,

        #     norm = self.norm_level/self.nsamples,
            
        #     spskr = spskr_coeff * (self.var_level / self.err_level).sqrt(),

        #     unbiased_spskr = (self.norm_level / self.prederr_level).sqrt(),
            
        #     crps = (self.mae_level - .5*self.disp_level)/self.nsamples,
        #     )
        

        out = dict()
        for name, i, j in self.display_surface_metrics:
            out.update({name+'_'+k: v[i, j] for k, v in surface_metrics.items()})

        # for name, i, j in self.display_level_metrics:
        #     out.update({name+'_'+k: v[i, j] for k, v in level_metrics.items()})

        out['headline_unbiased_spskr'] = torch.stack([v for k, v in out.items() if 'unbiased_spskr' in k]).mean()
        out['headline_spskr'] = torch.stack([v for k, v in out.items() if 'spskr' in k and not 'unbiased' in k]).mean()


        return out
