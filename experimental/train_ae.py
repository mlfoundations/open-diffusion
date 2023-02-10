from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

class Autoencoder(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()
    
    