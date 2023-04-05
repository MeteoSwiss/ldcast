import dwd_dataset
import train_genforecast


def data_iter_mchrzc(resolution=256, batch_size=8, split="test"):
    sampler_root = "sampler_nowcaster"
    if resolution == 256:
        sampler_root += "256"
    sampler_file = {
        s: f"../cache/{sampler_root}_{s}.pkl" 
        for s in ["test", "train", "valid"]
    }

    sample_shape = (resolution//32,) * 2
    datamodule = train_genforecast.setup_data(
        batch_size=batch_size, use_nwp=False, 
        sample_shape=sample_shape, sampler_file=sampler_file
    )
    if split == "test":
        dataloader = datamodule.test_dataloader()
    elif split == "valid":
        dataloader = datamodule.val_dataloader()
    return iter(dataloader)


def data_iter_testset(**kwargs):
    return data_iter_mchrzc(**kwargs, split="test")


def data_iter_validset(**kwargs):
    return data_iter_mchrzc(**kwargs, split="valid")


def data_iter_dwdrv(resolution=256, batch_size=8):
    sampler_file = {"test": f"../cache/sampler_dwd.pkl"}
    datamodule = dwd_dataset.setup_data(
        batch_size=batch_size, use_nwp=False,
        sampler_file=sampler_file    
    )
    dataloader = datamodule.test_dataloader()
    return iter(dataloader)


def get_data_iter(dataset_id="testset", **kwargs):
    func = {
        "testset": data_iter_testset,
        "validset": data_iter_validset,
        "dwdrv": data_iter_dwdrv
    }[dataset_id]
    return func(**kwargs)
