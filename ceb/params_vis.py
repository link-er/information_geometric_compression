sample_size = 8    # number of Z to sample for each X
latent_size = 2        # Dimension of Z

dset_dir = "datasets/FashionMNIST"
input_size = 28 * 28
num_classes = 10

batch_size = 64              # batch size

data_loaders = get_dataloaders(dset_dir, batch_size)

layer_width = 256
num_layers_enc = 9
encoder_model = MLP(input_size, latent_size*2, num_layers_enc, layer_width).cuda()
decoder_model = MLP(latent_size, num_classes, 5, layer_width).cuda()
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