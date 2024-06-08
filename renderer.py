from config import *

def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    sampler: Sampler,
    x_start=None,
    x_ref=None,
    cond=None,
    mask=None
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x_start)['cond']
        return sampler.sample(model=model,
                              x_start=x_start,
                              x_ref=x_ref,
                              noise=x_T,
                              mask=mask,
                              model_kwargs={'cond': cond})
    else:
        raise NotImplementedError()
