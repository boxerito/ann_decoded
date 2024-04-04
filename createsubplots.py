import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of combinations of n and i
name=['RFs','cost_vs_epoch','orig_vs_reconstructed']
for name in name:
    for n in [250,500,1000,2000]:
        # Initialize the figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        for i, ax in zip([1000, 3000, 10000, 30000], axs.flatten()):
            # Generate the image file name based on n and i
            img_file = f'ANNpictures/{name}_{n}n_{i}img.png'
            
            # Load and display the image
            img = mpimg.imread(img_file)
            ax.imshow(img)
            ax.axis('off')
            fontsize=8
            # Set the title based on the name
            if name == 'RFs':
                ax.set_title(f'RFs ({n} neurons, {i} images)', fontsize=fontsize)
            elif name == 'cost_vs_epoch':
                ax.set_title(f'Cost vs Epoch ({n} neurons, {i} images)', fontsize=fontsize)
            elif name == 'orig_vs_reconstructed':
                ax.set_title(f'Original vs Reconstructed ({n} neurons, {i} images)', fontsize=fontsize)
            
        # Adjust the spacing between subplots
        
        plt.subplots_adjust(wspace=0, hspace=0.0)
        
        plt.savefig(f'ANNpictures/{name}for{n}neurons.png', dpi=500)
        plt.show()