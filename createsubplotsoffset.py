import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of combinations of n and i
name=['RFs','cost_vs_epoch','orig_vs_reconstructed']
n=1000
offset_scale = range(0,6)
for name in name:
    for i in [1000, 3000, 10000, 30000]:
        # Initialize the figure and subplots
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        
        for offset, ax in zip(offset_scale, axs.flatten()):
            # Generate the image file name based on n and i
            img_file = f'ANNpictures/{name}_{n}n_{i}img_{offset}offset.png'
            
            # Load and display the image
            img = mpimg.imread(img_file)
            ax.imshow(img)
            ax.axis('off')
            fontsize=8
            # if name == 'RFs':
            #     ax.set_title(f'RFs ({n} neurons,{i} images,{noise} noise)', fontsize=fontsize)
            # elif name == 'cost_vs_epoch':
            #     ax.set_title(f'Cost vs Epoch ({n} neurons,{i} images,{noise} noise)', fontsize=fontsize)
            # elif name == 'orig_vs_reconstructed':
            #     ax.set_title(f'Original vs Reconstructed ({n} neurons,{i} images,{noise} noise)', fontsize=fontsize)
            
        # Adjust the spacing between subplots
        if name == 'cost_vs_epoch':
            plt.subplots_adjust(wspace=0, hspace=0.0)
        else:
            plt.subplots_adjust(wspace=0, hspace=0)

        # Adjust the spacing between subplots
        plt.tight_layout()
        
        plt.savefig(f'ANNpictures/{name}for{n}neurons{i}img_offset.png', dpi=500)
        plt.show()