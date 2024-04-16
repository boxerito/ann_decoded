import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of combinations of n and i
name_list=['reconstruction','model_loss','First_Conv_Layer_Filters']
norm_scale=True
for name in name_list:
    for n in [250,500,1000,2000]:
        # Initialize the figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        for i, ax in zip([1000, 3000, 10000, 30000], axs.flatten()):
            # Generate the image file name based on n and i
            fname = f'neurons_to_cifar_{n}n_1rep{i}n_img'
            if norm_scale==True and name=='model_loss':
                img_file = f'ANNpictures/{name}_Autoencoder_{fname}_normscale.png'
            else:
                img_file = f'ANNpictures/{name}_Autoencoder_{fname}.png'
            # Load and display the image
            img = mpimg.imread(img_file)
            ax.imshow(img)
            ax.axis('off')
            fontsize=8
            # Set the title based on the name
            if name == 'reconstruction':
                ax.set_title(f'Orig. vs recons. ({n} neurons, {i} images)', fontsize=fontsize)
                plt.subplots_adjust(wspace=0, hspace=-0.8)
            elif name == 'model_loss':
                ax.set_title(f'Cost vs Epoch ({n} neurons, {i} images)', fontsize=fontsize)
                plt.subplots_adjust(wspace=0, hspace=0)
            elif name == 'First_Conv_Layer_Filters':
                ax.set_title(f'First Conv. Layer Filters ({n} neurons, {i} images)', fontsize=fontsize)
                plt.subplots_adjust(wspace=0, hspace=-0.8)
        

        plt.tight_layout()
        if norm_scale==True and name=='model_loss':
            plt.savefig(f'ANNpictures/{name}for{n}neurons_Autoencoder_normscale.png', dpi=500)
        else:
            plt.savefig(f'ANNpictures/{name}for{n}neurons_Autoencoder.png', dpi=500)
        plt.show()