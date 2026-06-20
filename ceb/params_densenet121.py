sample_size = 8    # number of Z to sample for each X
latent_size = 256        # Dimension of Z

dset_dir = "datasets/CIFAR100"
num_classes = 100

batch_size = 256              # batch size

data_loaders = get_dataloaders(dset_dir, batch_size)

layer_width = 1024
# Densenet 121
encoder_model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=latent_size*2).cuda()
decoder_model = MLP(latent_size, num_classes).cuda()
backencoder_model = MLP(num_classes, latent_size*2).cuda()

# Build the model:
model = CEB(beta=args.beta,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            latent_size=latent_size,
            backencoder_model=backencoder_model,
            num_classes=num_classes,
            sample_size=sample_size
           )