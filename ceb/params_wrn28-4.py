sample_size = 8    # number of Z to sample for each X
latent_size = 256        # Dimension of Z

dset_dir = "datasets/CIFAR10"
num_classes = 10

batch_size = 256              # batch size

data_loaders = get_dataloaders(dset_dir, batch_size)

layer_width = 1024
encoder_model = WideResNet(28, latent_size*2, widen_factor=4).cuda()
decoder_model = MLP(latent_size, num_classes, 0, layer_width).cuda()
backencoder_model = MLP(num_classes, latent_size*2, 0, 0).cuda()

# Build the model:
model = CEB(beta=args.beta,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            latent_size=latent_size,
            backencoder_model=backencoder_model,
            num_classes=num_classes,
            sample_size=sample_size
           )