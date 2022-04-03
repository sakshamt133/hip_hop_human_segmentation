from dataset import HipHop
import utils
from torch.utils.data import DataLoader, random_split

all_data = HipHop(utils.img_path, utils.mask_path, utils.transform)
train_size = int(0.9 * len(all_data))
test_size = len(all_data) - train_size

train_data, test_data = random_split(all_data, [train_size, test_size])

train_batch = DataLoader(
    train_data,
    batch_size=utils.batch_size,
    shuffle=True,
    num_workers=8
)

test_batch = DataLoader(
    test_data,
    batch_size=utils.batch_size,
    num_workers=8
)