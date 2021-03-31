import os

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, str(int(self.num_timesteps/self.save_freq)).zfill(5))
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True