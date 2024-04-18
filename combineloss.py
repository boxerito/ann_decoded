# Description: This file contains the functions to calculate the AUC of a model
#call this function before model fitting
def extract_and_combine_losses(model, x_train, y_train, x_test, y_test, stage, history=None):
    if stage=="pre": #pretraining
        #extract the initial loss of training
        initial_loss_train = model.evaluate(x_train, y_train, verbose=0)
        print(f'Pérdida inicial en el conjunto de entrenamiento: {initial_loss_train}')
        #extract the initial loss of validation
        initial_loss_val = model.evaluate(x_test, y_test, verbose=0)
        print(f'Pérdida inicial en el conjunto de validación: {initial_loss_val}')
        return initial_loss_train, initial_loss_val
    elif stage=="post": #training
        if history is None:
            print("Error: history is required for 'training' stage")
            return None
        combined_loss_train = []
        combined_loss_train.extend(history.history['loss'])
        #and the final loss
        final_loss_train = model.evaluate(x_train, y_train, verbose=0)
        print(f'Pérdida final en el conjunto de entrenamiento: {final_loss_train}')
        #append it
        combined_loss_train.append(final_loss_train)
        
        combined_loss_val = []
        combined_loss_val.extend(history.history['val_loss'])
        #and the final loss
        final_loss_val = model.evaluate(x_test, y_test, verbose=0)
        print(f'Pérdida final en el conjunto de validación: {final_loss_val}')
        #append it
        combined_loss_val.append(final_loss_val)
        return combined_loss_train, combined_loss_val
    else:
        print("Error: stage should be either 'pre' or 'post'")
        return None
#let's define now a function to combine both pretraining a training losses
def combine_pretraining_and_training_losses(initial_combined_loss,combined_loss,plot=None):
    #combine both losses
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    initial_loss_train = initial_combined_loss[0]
    initial_loss_val = initial_combined_loss[1]
    combined_loss_train = combined_loss[0]
    combined_loss_val = combined_loss[1]
    combined_loss_train = [initial_loss_train] + combined_loss_train
    combined_loss_val = [initial_loss_val] + combined_loss_val
    factornorm = max([max(combined_loss_train),max(combined_loss_val)])
    combined_loss_train_norm = np.array(combined_loss_train)/factornorm
    combined_loss_val_norm = np.array(combined_loss_val)/factornorm
    
    if plot==None:
        #open new figure
        fig, ax = plt.subplots()
        epochs=range(len(combined_loss_train))
        plt.plot(epochs,combined_loss_train_norm, label='train loss (normalized)',color='red')
        plt.plot(epochs,combined_loss_val_norm, label='validation loss (normalized)',color='blue')
        plt.ylim(0, 1)
        plt.xlim(0, len(combined_loss_train) - 1)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% of Initial Loss')
        plt.title('Training and Validation Loss (Normalized)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Normalized by Initial Max. Loss)')
        # plt.legend()
        # plt.show()
        def exp_decay(epoch, L0, alpha, Lmin):
            return L0 * np.exp(-alpha * epoch) + Lmin
        # Ajuste de la curva
        try: 
            params, _ = curve_fit(exp_decay, epochs, combined_loss_val_norm)
            L0, alpha, Lmin = params
            print(f'L0: {L0}, alpha: {alpha}, Lmin: {Lmin}')
            # Plot the exponential decay curve
            x_points=np.linspace(0,len(combined_loss_train)-1)
            plt.plot(x_points, exp_decay(x_points, L0, alpha, Lmin), 'k--', label='Exp. Decay Fit-Val Loss')
            #make these variables global
            global x_text, y_text
            x_text=0.5
            y_text=0.7
            plt.text(x_text, y_text, f'$L(E) = {L0:.3f} \cdot e^{{-{alpha:.3f} \cdot Epoch}} + {Lmin:.3f}$', transform=plt.gca().transAxes)
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            # Asumiendo que y_true son tus valores reales y y_pred son las predicciones de tu modelo
            y_true = combined_loss_val_norm
            y_pred = exp_decay(epochs, L0, alpha, Lmin)
            #print MSE below the equation
            errormethod = 'MAE'
            if errormethod=='MSE':
                mse = mean_squared_error(y_true, y_pred)
                plt.text(x_text, y_text-0.05, f'MSE: {mse:.3E}', transform=plt.gca().transAxes)

            elif errormethod=='MAE':
                mae = mean_absolute_error(y_true, y_pred)
                plt.text(x_text, y_text-0.05, f'MAE: {mae:.3E}', transform=plt.gca().transAxes)
            elif errormethod=='LogRMSE' or errormethod=='logrmse':
                log_rmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
                plt.text(x_text, y_text-0.05, f'LogRMSE: {log_rmse:.3E}', transform=plt.gca().transAxes)
            else:
                print("Error: errormethod should be either 'MSE', 'MAE' or 'LogRMSE'")
                return None

        except:
            print("Error: curve_fit failed")
            plt.text(x_text, y_text, f'Error: curve_fit failed', transform=plt.gca().transAxes)
            # raise ValueError("Error: curve_fit failed") #raise an error in case of failure
        plt.legend()
        plt.show()
    elif plot==False:
        pass
    else:
        print("Error: plot should be either None or False")
        return None
    return combined_loss_train, combined_loss_val, fig