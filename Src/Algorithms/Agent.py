from math import exp

# Parent to all algorithm
class Agent:

    def __init__(self, config):
        self.config = config

        # External/no-purchase option.  Pricing formulas in this codebase use a
        # normalized outside option with utility 0; when external_option=True we
        # replace that normalization with a configurable outside utility and the
        # environment also lets customers leave without choosing a delivery option.
        self.external_option = bool(getattr(config, "external_option", False))
        self.external_base_util = float(getattr(config, "external_base_util", 0.0))
        self.external_price = float(getattr(config, "external_price", 0.0))
        external_alpha = getattr(config, "external_price_sensitivity", None)
        if external_alpha is None:
            # Match U_external = base_external - alpha * price and, by default,
            # use the absolute value of the existing negative price coefficient.
            external_alpha = max(-float(getattr(config, "incentive_sens", 0.0)), 0.0)
        self.external_price_sensitivity = float(external_alpha)

        # Abstract class variables
        self.modules = None

    def get_external_utility(self):
        """Return U_external = base_external - alpha_external * price_external."""
        return self.external_base_util - self.external_price_sensitivity * self.external_price

    def get_external_mnl_weight(self, for_pricing=True):
        """Return exp(U_external) with safe clipping.

        When the explicit external option is disabled, pricing keeps the legacy
        normalized outside-option scale of 1.0, while realized choice models use
        0.0 so probabilities still sum over the offered delivery options only.
        """
        if not self.external_option:
            return 1.0 if for_pricing else 0.0
        utility = max(min(float(self.get_external_utility()), 700.0), -700.0)
        return exp(utility)

    def adjust_lambert_sum_for_external_option(self, sum_mnl):
        """Scale the Lambert-W pricing term by the external-option MNL weight."""
        external_weight = max(self.get_external_mnl_weight(for_pricing=True), 1e-300)
        return sum_mnl / external_weight

    def init(self):
         for name, m in self.modules:
             m.to(self.config.device)

    def clear_gradients(self):
        for _, module in self.modules:
            module.optim.zero_grad()
            
    def clear_actor_gradients(self):
        self.modules[0][1].optim.zero_grad()

    def clear_critic_gradients(self):
        self.modules[1][1].optim.zero_grad()

    def save(self):
        if self.config.save_model:
            for name, module in self.modules:
                module.save(self.config.paths['checkpoint'] + name+'.pt')

    def step(self, loss, clip_norm=False):
        self.clear_gradients()
        loss.backward()
        for _, module in self.modules:
            module.step(clip_norm)
            
    def actor_step(self, loss, clip_norm=False):
        self.clear_actor_gradients()
        loss.backward()
        self.modules[0][1].step(clip_norm)
    def critic_step(self, loss, clip_norm=False):
        self.clear_actor_gradients()
        loss.backward()
        self.modules[1][1].step(clip_norm)

    def reset(self):
        for _, module in self.modules:
            module.reset()
