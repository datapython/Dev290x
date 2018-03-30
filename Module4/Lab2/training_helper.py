import numpy as np
import cntk_resnet_fcn
import cntk as C
from cntk.learners import learning_rate_schedule, UnitType

#-------------------------------------------------------------------------------
def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]
    
    return sx, sy

#-------------------------------------------------------------------------------
def measure_error(source, data_x_files, data_y_files, x, y, trainer, minibatch_size):
    errors = []
    for i in range(0, int(len(data_x_files) / minibatch_size)):
        data_sx_files, data_sy_files = slice_minibatch(data_x_files, data_y_files, i, minibatch_size)
        data_sx, data_sy = source.files_to_data(data_sx_files, data_sy_files)
        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)

#-------------------------------------------------------------------------------

def train(train_images, train_masks, val_images, val_masks, base_model_file, freeze=False):
    shape = train_images[0].shape
    data_size = train_images.shape[0]

    test_data = (val_images, val_masks)
    training_data = (train_images, train_masks)

    # Create model
    x = C.input_variable(shape)
    y = C.input_variable(train_masks[0].shape)
    
    z = cntk_resnet_fcn.create_transfer_learning_model(x, train_masks.shape[1], base_model_file, freeze)
    dice_coef = cntk_resnet_fcn.dice_coefficient(z, y)

    # Prepare model and trainer
    lr_mb = [0.0001] * 10 + [0.00001]*10 + [0.000001]*10 + [0.0000001]*10
    lr = learning_rate_schedule(lr_mb, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0.9)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 8
    num_epochs = 60
    
    training_errors = []
    test_errors = []

    for e in range(0, num_epochs):
        for i in range(0, int(len(training_data[0]) / minibatch_size)):
            data_x, data_y = slice_minibatch(training_data[0], training_data[1], i, minibatch_size)
            trainer.train_minibatch({z.arguments[0]: data_x, y: data_y})
        
        # Measure training error
        training_error = measure_error(training_data[0], training_data[1], z.arguments[0], y, trainer, minibatch_size)
        training_errors.append(training_error)
        
        # Measure test error
        test_error = measure_error(test_data[0], test_data[1], z.arguments[0], y, trainer, minibatch_size)
        test_errors.append(test_error)

        print("epoch #{}: training_error={}, test_error={}".format(e, training_errors[-1], test_errors[-1]))
        
    return trainer, training_errors, test_errors


