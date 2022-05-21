import torch


class Base:
    def __call__(self, *args, return_numpy=True, **kwargs):
        tensors = self.net(*args, **kwargs)

        if return_numpy:

            #  handle if we have one parameter
            if len(tensors) > 1:
                return [tensor.detach().numpy() for tensor in tensors]
            else:
                return tensors.detach().numpy()

        else:
            return tensors

    def save(
        self,
        net_path,
        opt_path
    ):
        torch.save(
            self.net.state_dict(),
            net_path.with_suffix('.pth')
        )
        torch.save(
            self.optimizer.state_dict(),
            opt_path.with_suffix('.pth')
        )
