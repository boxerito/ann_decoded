import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of combinations of n and i
name_list=['reconstruction','model_loss']
for name in name_list:
    for n in [250,500,1000,2000]:
        # Initialize the figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        for i, ax in zip([1000, 3000, 10000, 30000], axs.flatten()):
            # Generate the image file name based on n and i
            fname = f'neurons_to_cifar_{n}n_1rep{i}n_img'
            img_file = f'ANNpictures/{name}_MLP_{fname}.png'
            
            # Load and display the image
            img = mpimg.imread(img_file)
            ax.imshow(img)
            ax.axis('off')
            fontsize=8
            # Set the title based on the name
            if name == 'reconstruction':
                ax.set_title(f'Orig. vs recons. ({n} neurons, {i} images)', fontsize=fontsize)
                plt.subplots_adjust(wspace=0, hspace=-0.5)
            elif name == 'model_loss':
                ax.set_title(f'Cost vs Epoch ({n} neurons, {i} images)', fontsize=fontsize)
                plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()

        plt.savefig(f'ANNpictures/{name}for{n}neurons_MLP.png', dpi=500)
        plt.show()